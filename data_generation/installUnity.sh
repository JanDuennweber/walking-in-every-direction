#!/bin/bash
# Installs Unity Hub from Unity's official RPM repository (targets
# RHEL/CentOS; Fedora is not officially supported but uses the same RPMs).
# Needed to generate the row 3/4 training data for Table 3, see
# README.md.
#
# Usage: sudo ./installUnity.sh
#
# Afterwards (as your normal user, NOT with sudo):
#   1. run `unityhub`, sign in, activate a free Personal license
#   2. install the Editor version the project pins:
#        unityhub -- --headless install --version 2022.3.26f1 --changeset ec6cd8118806
set -e

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo: sudo $0" >&2
  exit 1
fi

tee /etc/yum.repos.d/unityhub.repo > /dev/null <<'EOF'
[unityhub]
name=Unity Hub
baseurl=https://hub.unity3d.com/linux/repos/rpm/stable
enabled=1
gpgcheck=1
gpgkey=https://hub.unity3d.com/linux/repos/rpm/stable/repodata/repomd.xml.key
repo_gpgcheck=1
EOF

dnf install -y unityhub

echo ""
echo "Unity Hub installed. Next steps (as your normal user, not root):"
echo "  1. unityhub   -> sign in, activate a Personal license"
echo "  2. unityhub -- --headless install --version 2022.3.26f1 --changeset ec6cd8118806"
