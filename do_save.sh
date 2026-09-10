#!/usr/bin/env bash

# Stop at first error
set -e
# AND at the first error inside a pipe. Without this, `docker save | gzip > f`
# reports success whenever gzip exits 0 -- which it does even when docker save
# died half-way, having written a valid-prefix but TRUNCATED file. That is how
# the sixth submission's 3.3 GB image tarball shipped as 1.26 GB of unreadable
# gzip while this script printed "Container image saved".
set -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# DOCKER_IMAGE_TAG is set by do_build.sh, which is sourced below. Declaring it
# here would be dead: the source overwrites it. (It used to say "Submission
# algoritm category 1 final phase" -- typo, spaces, and never the tag that was
# actually built or saved.)
echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

# Get the build information from the Docker image tag
build_timestamp=$( docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")

if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for container $DOCKER_IMAGE_TAG"
    exit 1
fi

# Format the build information to remove special characters
formatted_build_info=$(echo $build_timestamp | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')

# Set the output filename with timestamp and build information
output_filename="${SCRIPT_DIR}/${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"

# Save the Docker-container image and gzip it
echo "==+=="
echo "Saving the container image as ${output_filename}. This can take a while."

echo ""

docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$output_filename"

# Read it back. `pipefail` catches a failed producer, but not a full disk that
# silently short-writes, and this file is the artefact that gets uploaded: the
# only trustworthy check is decompressing it.
echo "Verifying ${output_filename} ..."
gzip -t "$output_filename" || {
    echo "ERROR: ${output_filename} does not decompress -- it is truncated or corrupt." >&2
    echo "       Do NOT upload it. Check free disk and re-run." >&2
    exit 1
}
echo "Container image saved and verified as ${output_filename}"
echo "==+=="

# Create the tarbal
echo "==+=="
output_tarball_name="${SCRIPT_DIR}/nmodel.tar.gz"
echo "Creating the optional tarball as ${output_tarball_name}. This can take a while."

echo ""

# whole_instrument/ is excluded on purpose: inference.py globs ONLY
# model/ensemble/*.pth (see its load_models comment -- those detectors find the
# WHOLE instrument, a different object, and enrolling them would flood every
# frame with instrument-sized boxes labelled as clevises). Shipping them was 250
# MB, 44% of this tarball, that the container never opens.
tar -czf $output_tarball_name --exclude=./whole_instrument -C "${SCRIPT_DIR}/model" .
gzip -t "$output_tarball_name" || {
    echo "ERROR: ${output_tarball_name} does not decompress." >&2; exit 1; }
echo "(Optional) Uploadable tarball was created and verified as ${output_tarball_name}"
echo "==+=="
