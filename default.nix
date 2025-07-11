{
  pkgs ? import <nixpkgs> { },
  ...
}:
let
  inherit (pkgs)
    lib
    cmake
    pkg-config
    sdl3
    sdl3-ttf
    ;
in
pkgs.stdenv.mkDerivation {
  pname = "brain-ager";
  version = "0.1.0";

  src = lib.cleanSource ./.;

  nativeBuildInputs = [
    cmake
    pkg-config
    sdl3
    sdl3-ttf
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DTESTING_BUILD=on"
  ];

  enableParallelBuilding = true;
  doCheck = true;

  meta = with lib; {
    description = "A neural network-powered math game.";
    homepage = "https://github.com/CS1103/projecto-final-turinmachin";
    license = licenses.gpl3;
    mainProgram = "brain_ager";
  };
}
