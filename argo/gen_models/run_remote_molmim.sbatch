#!/bin/bash
#SBATCH --job-name=molmim_srv
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=molmim_srv_%j.log

module load apptainer
export NGC_CLI_API_KEY=$APPTAINER_DOCKER_PASSWORD

WORKDIR=/work/users/s/h/shuhang/bionemo
FAKE_HOME=${WORKDIR}/fake_home_override
CACHE_DIR=${WORKDIR}/bionemo_cache
TMP_WORK=${WORKDIR}/container_tmp
APPTAINER_TMPDIR=${WORKDIR}/apptainer_tmp
export APPTAINER_TMPDIR

mkdir -p "$FAKE_HOME/.ngc" "$CACHE_DIR" "$TMP_WORK" "$APPTAINER_TMPDIR"

# Write NGC config inside fake home
cat > "${FAKE_HOME}/.ngc/config" <<EOF
[ngc]
apiKey=$NGC_CLI_API_KEY
EOF

echo "=== paths ==="
echo "WORKDIR   : $WORKDIR"
echo "FAKE_HOME : $FAKE_HOME"
echo "CACHE_DIR : $CACHE_DIR"
echo "TMP_WORK  : $TMP_WORK"
echo "APT TMP   : $APPTAINER_TMPDIR"
echo "==========="

# --------------------------------------------------
# Run MolMIM NIM server
apptainer exec \
  --nv \
  --cleanenv \
  --containall \
  --no-home \
  --env NGC_CLI_API_KEY=$NGC_CLI_API_KEY \
  --env TMPDIR=/tmp \
  --mount type=bind,source=$FAKE_HOME,destination=/nas/longleaf/home/$USER \
  --bind $CACHE_DIR:/home/nvs/.cache/nim \
  --bind $TMP_WORK:/tmp \
  /proj/kpoplab/molmim/molmim_1.0.0.sif start_server --port 8000

