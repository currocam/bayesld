using Makie, CairoMakie
using NPZ

# Load data
data = npzread("analytical_data_means.npz")
midpoints = (data["constant_left"] .+ data["constant_right"]) ./ 2 .* 100
times = 0:0.1:100

with_theme(theme_latexfonts()) do
    f = Figure(fontsize = 12)
    
    # Use default color palette
    colors = Makie.wong_colors()
    color_constant = colors[1]
    color_growth = colors[2]
    
    ax1 = Axis(f[1, 1],
        xlabel = "Time (generations ago)",
        ylabel = "Effective population size",
        subtitle="Demographic scenario"
    )
    
    stairs!(ax1, times, 10_000 * exp.(0.0 .* times), 
            color = color_constant, 
            linewidth = 2, step = :post)
    # Growth rate is positive
    stairs!(ax1, times, 10_000 * exp.(-(0.05) .* times), 
            color = color_growth, 
            linewidth = 2, step = :post)
    
    xlims!(ax1, 0, 50)
    ylims!(ax1, 0, 11_000)
    
    ax2 = Axis(f[1, 2],
        xlabel = "Distance (centiMorgan)",
        ylabel = L"\mathbb{E}[X_i Y_i X_j Y_j]",
        subtitle="Linkage disequilibrium decay"
    )
    
    scatter!(ax2, midpoints, data["constant_sims_mean"], 
             color = color_constant, markersize = 6)
    lines!(ax2, midpoints, data["constant_predictions_mean"], 
           color = color_constant, linewidth = 2)
    
    scatter!(ax2, midpoints, data["growth_sims_mean"], 
             color = color_growth, markersize = 6)
    lines!(ax2, midpoints, data["growth_predictions_mean"], 
           color = color_growth, linewidth = 2)
    
    # Custom legend outside the plots
    Legend(f[2, :],
        [
            [LineElement(color = color_constant, linewidth = 2),
             LineElement(color = color_growth, linewidth = 2)],
            [MarkerElement(color = color_constant, marker = :circle, markersize = 10),
             LineElement(color = color_constant, linewidth = 2)]
        ],
        [
            ["Constant", "Growth"],
            ["Simulations", "Predictions"]
        ],
        ["Scenario", "Data type"],
        orientation = :horizontal,
        tellwidth = false,
        tellheight = true,
        titleposition = :left
    )
    
    Label(f[1, 1, Bottom()], "A", fontsize = 14, halign = :left)
    Label(f[1, 2, Bottom()], "B", fontsize = 14, halign = :left)
    
    save("figure.pdf", f, pt_per_unit = 1)
    f
end




