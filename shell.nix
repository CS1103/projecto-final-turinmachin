{
  pkgs ? import <nixpkgs> { },
  ...
}:
let
  inherit (pkgs)
    mkShell
    SDL2
    SDL2_ttf
    SDL2_gfx
    catch2_3
    ;
in
mkShell {
  inputsFrom = [ (pkgs.callPackage ./default.nix { }) ];

  env = {
    CMAKE_PREFIX_PATH = "${SDL2.dev}/lib/cmake:${SDL2_ttf}/lib/cmake:${catch2_3}/lib/cmake";
    PKG_CONFIG_PATH = "${SDL2_gfx}/lib/pkgconfig";
  };
}
