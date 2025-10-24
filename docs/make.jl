using Onion
using Documenter

DocMeta.setdocmeta!(Onion, :DocTestSetup, :(using Onion); recursive=true)

makedocs(; 
    modules=[Onion, Onion.UNet, Onion.Utils],
    authors="murrellb <murrellb@gmail.com> and contributors",
    sitename="Onion.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/Onion.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Overview" => "index.md",
        "API Reference" => "API.md"
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/Onion.jl",
    devbranch="main",
)
