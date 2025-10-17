using NPZ
data = npzread("data.npz")
# Load data
using CairoMakie

with_theme(theme_latexfonts()) do
    # Set figure width
    fig = Figure(size = (379.4175, 200) .* 2, fontsize = 14)

    # Create two side-by-side axes
    ax1 = Axis(
        fig[1, 2],
        yscale = log10,
        xlabel = L"\text{Distance (centimorgan)}",
        ylabel = L"\mathbb{E}[X_iY_iX_jY_j]",
        subtitle = "Linkage disequilibrium decay"
    )

    ax2 = Axis(
        fig[1, 1],
        xlabel = L"\text{Time (generations ago)}",
        ylabel = L"\text{Effective population size (Ne)}",
        subtitle = "Demographic scenario"
    )

    # Define line styles (using default Makie color palette)
    linestyles = [:solid, :dash, :dot, :dashdot]

    # Constants from the simulations
    Nec = 5_000
    t_inv = 50
    times = 0:70

    # Store plot elements for legend
    elements = []
    labels = []

    # Plot for each index
    for (idx, i) in enumerate([1, 3, 4, 5])
        alpha = data["alphas"][i]

        # Left plot: scatter data
        s = scatter!(ax1, data["midpoints"], data["predictions"][i, :])

        # Right plot: population size over time
        Nea = Nec * exp(-alpha * t_inv)
        y = Nec .* exp.(-alpha * times)
        y[times .>= t_inv] .= Nea

        l = lines!(
            ax2, times, y,
            linestyle = linestyles[idx],
            linewidth = 3
        )

        # Store for legend (use line element with marker)
        push!(elements, [s, l])
        push!(labels, "α=$(alpha)")
    end

    # Add shared legend below, centered across both axes
    Legend(
        fig[2, :], elements, labels,
        orientation = :horizontal,
        tellwidth = false,
        tellheight = true,
        halign = :center
    )
    save("figure.pdf", fig, pt_per_unit = 1)
    fig
end
