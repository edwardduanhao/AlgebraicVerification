module EuclideanHC

include("Utils.jl")
using .Utils
using HomotopyContinuation
using ProgressBars
using NPZ

export euclidean_hc, robust_radius

function euclidean_hc(f, x, xi::Vector{Float64}, lmbda, verbose::Bool=false)
    # `lmbda` is the Lagrange multiplier polyvar; created once per run by the
    # caller so we don't keep growing DynamicPolynomials' variable counter.
    L = sum((x .- xi) .^ 2) + lmbda * f
    ∇L = differentiate(L, [x; lmbda])
    F = System(∇L; variables=[x; lmbda])
    sols = solve(F, show_progress=verbose)

    # Get only real solutions
    realsols = real_solutions(sols)

    # Find the closest real solution
    min_distance = Inf
    closest_sol = nothing
    closest_idx = 0

    for (i, sol) in enumerate(realsols)
        x_sol = sol[1:length(x)]  # Extract x values (exclude λ)
        λ_sol = sol[end]  # Extract λ value
        distance = sqrt(sum((x_sol .- xi) .^ 2))

        # Track the closest solution
        if distance < min_distance
            min_distance = distance
            closest_sol = x_sol
            closest_idx = i
        end
    end

    if verbose
        # Report the closest solution
        if !isnothing(closest_sol)
            println("Total number of solutions: ", length(sols))
            println("Number of real solutions: ", length(realsols))
            println("Closest x = ", closest_sol)
            println("Minimum distance: ", min_distance)
        else
            println("No real solutions found.")
        end
    end

    return closest_sol, min_distance, length(realsols), length(sols), realsols, sols
end


function robust_radius(project_root::String, xi_list::Vector{Vector{Float64}};
    verbose::Bool=false, save_path::Union{String,Nothing}=nothing,
    save_detailed::Bool=false)

    # Time model loading (includes weight loading + symbolic polynomial construction)
    timed_load = @timed Utils.load_model(project_root)
    model_forward, _ = timed_load.value
    model_load_wall_s = timed_load.time
    model_load_compile_s = timed_load.compile_time
    F, x = model_forward

    # Create the Lagrange-multiplier polyvar exactly once for this run.
    # Defining it inside euclidean_hc per call would grow DynamicPolynomials'
    # global variable counter for every (xi, class-pair) pair.
    @polyvar lmbda

    num_classes = length(F)
    robust_radii = Vector{Tuple{Float64,Any}}()
    detailed_results = []  # Store detailed results if requested

    # Allocate per-instance timing arrays
    n_points = length(xi_list)
    instance_wall_s = zeros(Float64, n_points)
    instance_compile_s = zeros(Float64, n_points)

    println("\n" * repeat("=", 60))
    println("Computing robust radius for ", n_points, " points")
    println("Julia threads: ", Threads.nthreads())
    println("Model load: $(round(model_load_wall_s, digits=3))s wall, $(round(model_load_compile_s, digits=3))s compile")

    for (idx, xi) in ProgressBar(enumerate(xi_list))
        timed_instance = @timed begin
            # Evaluate F at xi to determine the predicted class
            F_values = [f(x => xi) for f in F]
            k = argmax(F_values)

            # Compute minimum distance across all other classes
            min_dist_local = Inf
            closest_sol_local = nothing

            # Store detailed info for this point if requested
            point_details_local = if save_detailed
                Dict(
                    "xi" => xi,
                    "predicted_class" => k,
                    "boundary_pairs" => Vector{Tuple{Int,Int}}(),
                    "boundary_distances" => Vector{Float64}(),
                    "boundary_closest_sols" => Vector{Any}(),
                    "boundary_real_sols" => Vector{Any}(),
                    "boundary_num_real" => Vector{Int}(),
                    "boundary_num_total" => Vector{Int}()
                )
            else
                nothing
            end

            for l in 1:num_classes
                if l == k
                    continue
                end

                f = F[k] - F[l]

                sol, distance, num_real, num_total, realsols, _ = euclidean_hc(f, x, xi, lmbda, verbose)

                # Track the global minimum
                if !isnothing(sol) && distance < min_dist_local
                    min_dist_local = distance
                    closest_sol_local = sol
                end

                # Store detailed results for this boundary
                if save_detailed
                    push!(point_details_local["boundary_pairs"], (k, l))
                    push!(point_details_local["boundary_distances"], isnothing(sol) ? Inf : distance)
                    push!(point_details_local["boundary_closest_sols"], isnothing(sol) ? fill(NaN, length(xi)) : sol)
                    push!(point_details_local["boundary_real_sols"], realsols)
                    push!(point_details_local["boundary_num_real"], num_real)
                    push!(point_details_local["boundary_num_total"], num_total)
                end
            end

            (min_dist_local, closest_sol_local, point_details_local)
        end

        min_dist, closest_sol, point_details = timed_instance.value
        instance_wall_s[idx] = timed_instance.time
        instance_compile_s[idx] = timed_instance.compile_time

        push!(robust_radii, (min_dist, closest_sol))

        if save_detailed && !isnothing(point_details)
            push!(detailed_results, point_details)
        end

        if verbose
            println("Point $idx (xi=$xi): robust radius = $(min_dist), time = $(round(timed_instance.time, digits=3))s (compile: $(round(timed_instance.compile_time, digits=3))s)")
        end

        # Incremental collection between points keeps HC path-tracker allocations
        # from piling up across long xi_list sweeps.
        GC.gc(false)
    end

    # Save results if save_path is provided
    if !isnothing(save_path)
        # Extract min_dist and closest_sol arrays
        min_dists = [r[1] for r in robust_radii]

        # Convert xi_list to matrix (each row is a point)
        xi_matrix = hcat(xi_list...)' |> Array{Float64}

        # Handle closest_sol which may contain nothing values
        # Convert to matrix, using NaN for nothing values
        dim = length(xi_list[1])
        closest_sols_matrix = zeros(Float64, length(robust_radii), dim)
        for (i, r) in enumerate(robust_radii)
            if !isnothing(r[2])
                closest_sols_matrix[i, :] .= r[2]
            else
                closest_sols_matrix[i, :] .= NaN
            end
        end

        # Save as NPZ file (readable by numpy in Python)
        npzwrite(save_path, Dict(
            "xi_list" => xi_matrix,
            "min_dist" => min_dists,
            "closest_sol" => closest_sols_matrix
        ))

        # Save timing data to separate file
        timing_path = joinpath(dirname(save_path), "timing.npz")
        npzwrite(timing_path, Dict(
            "model_load_wall_s" => model_load_wall_s,
            "model_load_compile_s" => model_load_compile_s,
            "instance_wall_s" => instance_wall_s,
            "instance_compile_s" => instance_compile_s,
            "n_threads" => Threads.nthreads()
        ))

        # Use relative path for cleaner output
        display_path = try
            relpath(save_path)
        catch
            save_path
        end
        println("Results saved to: $display_path")

        display_timing_path = try
            relpath(timing_path)
        catch
            timing_path
        end
        println("Timing saved to: $display_timing_path")

        # Save detailed results if requested
        if save_detailed && !isempty(detailed_results)
            # Create detailed results directory
            save_dir = dirname(save_path)
            detailed_dir = joinpath(save_dir, "hc_detailed")
            mkpath(detailed_dir)

            for (i, details) in enumerate(detailed_results)
                point_file = joinpath(detailed_dir, "point_$(lpad(i-1, 3, '0')).npz")

                # Convert boundary_pairs to matrix for saving
                pairs_matrix = hcat([collect(p) for p in details["boundary_pairs"]]...)' |> Array{Int}

                # Convert boundary_closest_sols to matrix
                closest_sols_matrix = hcat(details["boundary_closest_sols"]...)' |> Array{Float64}

                # Convert boundary_real_sols: save each boundary's solutions separately
                # Each boundary can have different number of solutions
                save_dict = Dict(
                    "xi" => details["xi"],
                    "predicted_class" => details["predicted_class"],
                    "boundary_pairs" => pairs_matrix,
                    "boundary_distances" => details["boundary_distances"],
                    "boundary_closest_sols" => closest_sols_matrix,
                    "boundary_num_real" => details["boundary_num_real"],
                    "boundary_num_total" => details["boundary_num_total"]
                )

                # Add real solutions for each boundary
                for (j, realsols) in enumerate(details["boundary_real_sols"])
                    if !isempty(realsols)
                        # Convert to matrix: each row is a solution
                        sols_matrix = hcat(realsols...)' |> Array{Float64}
                        save_dict["boundary_real_sols_$(j-1)"] = sols_matrix
                    end
                end

                npzwrite(point_file, save_dict)
            end

            display_detailed_dir = try
                relpath(detailed_dir)
            catch
                detailed_dir
            end
        end
    end

    return robust_radii
end

# Overload for matrix input (convenient for Python interop)
function robust_radius(project_root::String, xi_mat::AbstractMatrix;
    verbose::Bool=false, save_path::Union{String,Nothing}=nothing,
    save_detailed::Bool=false)
    xi_list = [Vector{Float64}(xi_mat[i, :]) for i in axes(xi_mat, 1)]
    return robust_radius(project_root, xi_list; verbose=verbose, save_path=save_path, save_detailed=save_detailed)
end

end  # module EuclideanHC