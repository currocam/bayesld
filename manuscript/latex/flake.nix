{
  description = "LaTeX environment for rendering the manuscript";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        tex = pkgs.texlive.combine {
          inherit (pkgs.texlive)
            scheme-medium
            type1cm
            cm-super
            xifthen
            ifmtarg
            lineno
            mathpazo
            lettrine
            datetime
            fmtcount
            enumitem
            natbib
            lastpage
            microtype
            colortbl
            ragged2e
            algorithms
            algorithmicx
            changepage
            footmisc
            textpos
            titlesec
            preprint
            ;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [ tex ];
        };
      });
}
