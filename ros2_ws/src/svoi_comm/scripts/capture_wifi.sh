#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: IFACE=<wifi-interface> $0 OUTPUT.pcap"
  exit 2
fi

iface="${IFACE:-any}"
output="$1"
mkdir -p "$(dirname "$output")"
echo "Capturing UDP map/TF (12130) and reliable TCP goals (12140) on $iface"
exec sudo tcpdump -i "$iface" -s 0 -U -w "$output" \
  'udp port 12130 or tcp port 12140'
