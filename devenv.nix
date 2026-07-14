{
  pkgs,
  inputs,
  ...
}: {
  overlays = [
    (final: _prev: {
      stable = import inputs.nixpkgs-stable {
        inherit (final) config;
        system = pkgs.stdenv.hostPlatform.system;
      };
      unstable = import inputs.nixpkgs-unstable {
        inherit (final) config;
        system = pkgs.stdenv.hostPlatform.system;
      };
    })
  ];

  packages = with pkgs.unstable; [
    llvmPackages.openmp
    ruff
    pre-commit
  ];
  languages.python = {
    enable = true;
    package = pkgs.unstable.python312;
    uv = {
      enable = true;
      package = pkgs.unstable.uv;
    };
  };

  env.PYTORCH_ENABLE_MPS_FALLBACK = "1";
  env.DYLD_LIBRARY_PATH = "${pkgs.unstable.llvmPackages.openmp}/lib"; # Required by LightGBM
}
