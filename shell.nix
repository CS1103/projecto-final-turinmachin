{
  pkgs ? import <nixpkgs> { },
  ...
}:
let
  inherit (pkgs)
    mkShell
    sdl3
    sdl3-ttf
    ;
in
mkShell {
  inputsFrom = [ (pkgs.callPackage ./default.nix { }) ];

  env = {
    CMAKE_PREFIX_PATH = "${sdl3.dev}/lib/cmake:${sdl3-ttf}/lib/cmake";
  };
}
