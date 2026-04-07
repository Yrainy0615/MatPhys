#!/bin/bash
set -e

username=yyang
projectname=phys-gs

docker build -t repo-luna.ist.osaka-u.ac.jp:5000/${username}/${projectname}:build .
