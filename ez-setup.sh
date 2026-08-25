#!/bin/bash

set -e

echo "yoo, hows everything in chidzz"
echo "this is lil setup script - tailscale + ssh into ur dell -> Then it sshs into a100"

if ! command -v brew >/dev/null 2>&1; then
   echo "error - brew isnt found"
   echo "boss - install homebrew - dw il do it here itself"
   echo "installing brew"
   echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
   exit 1
fi

echo "lets install tailscale"
brew install --cask tailscale

echo "start that shit"
open -a Tailscale

echo "Login to tailscale - use google oauth and use ur sxdedegta one"
echo "press enter once ur login is complete g"
read

echo "check tailscale status"
for i in {1..30}; do
  if tailscale ip -4 >/dev/null 2>&1; then
     break
  fi
  sleep 2
done

echo "ur tailscale ip"
tailscale ip -4

echo "boss, call me for destination details - enter them below"
read -p "enter dell's tailscale ip:" DELL_IP
read -p "enter dell's username:" DELL_USERNAME

echo "lets connect"
ssh -t "${DELL_USERNAME}@${DELL_IP}"'
SERVER_USER="cs24d0010"
SERVER_HOST="172.16.1.199"

echo "connected to the jump nigga"
echo "connecting to the A100 now"

ssh -X ${SERVER_USER}@{SERVER_HOST}
'
