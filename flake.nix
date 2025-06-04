{
  description = "PINNs";

  inputs = {
    nixpkgs-unstable.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-25.05";
  };

  outputs = { self, nixpkgs, nixpkgs-unstable }: let
    pkgs-unstable = import nixpkgs-unstable {
      pure = true;
      system = "x86_64-linux";
      config = {
        allowUnfree = true;
        
      };
    };

    pkgs = import nixpkgs {
      pure = true;
      system = "x86_64-linux";
      config = {
        allowUnfree = true;
      };
    };

    pythonEnv = pkgs.python313.withPackages (ps: with ps; 
    [ 
      torch 
      numpy 
      ipython 
      scipy 
      einops 
      ipython 
      pip 
      utils 
      matplotlib 
      fenics 
    ]);


  in {
    packages.x86_64-linux = {
      default = pythonEnv;
    };

    devShell.x86_64-linux = pkgs.mkShell {
      pure = true;
      buildInputs = [ pkgs.bash pythonEnv ];

    };
  };
}