#!/usr/bin/env bash
#################################################
# Please do not make any changes to this file,  #
# change the variables in webui-user.sh instead #
#################################################

# If run from macOS, load defaults from webui-macos-env.sh
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ -f webui-macos-env.sh ]]
        then
        source ./webui-macos-env.sh
    fi
fi

# Read variables from webui-user.sh
# shellcheck source=/dev/null
if [[ -f webui-user.sh ]]
then
    source ./webui-user.sh
fi

# python3 executable
if [[ -z "${python_cmd}" ]]
then
    python_cmd="python3"
fi

if [[ -z "${LAUNCH_SCRIPT}" ]]
then
    LAUNCH_SCRIPT="launch.py"
fi

# Pretty print
delimiter="################################################################"

printf "\n%s\n" "${delimiter}"
printf "\e[1m\e[32mInstall script for stable-diffusion + Web UI\n"
printf "\e[1m\e[34mTested on Debian 11 (Bullseye)\e[0m"
printf "\n%s\n" "${delimiter}"

printf "\n%s\n" "${delimiter}"
printf "Repo already cloned, using it as install directory"
printf "\n%s\n" "${delimiter}"

# Try using TCMalloc on Linux
prepare_tcmalloc() {
    if [[ "${OSTYPE}" == "linux"* ]] && [[ -z "${NO_TCMALLOC}" ]] && [[ -z "${LD_PRELOAD}" ]]; then
        TCMALLOC="$(PATH=/usr/sbin:$PATH ldconfig -p | grep -Po "libtcmalloc(_minimal|)\.so\.\d" | head -n 1)"
        if [[ ! -z "${TCMALLOC}" ]]; then
            echo "Using TCMalloc: ${TCMALLOC}"
            export LD_PRELOAD="${TCMALLOC}"
        else
            printf "\e[1m\e[31mCannot locate TCMalloc (improves CPU memory usage)\e[0m\n"
        fi
    fi
}

KEEP_GOING=1
while [[ "$KEEP_GOING" -eq "1" ]]; do
    if [[ ! -z "${ACCELERATE}" ]] && [ ${ACCELERATE}="True" ] && [ -x "$(command -v accelerate)" ]; then
        printf "\n%s\n" "${delimiter}"
        printf "Accelerating launch.py..."
        printf "\n%s\n" "${delimiter}"
        prepare_tcmalloc
        accelerate launch --num_cpu_threads_per_process=6 "${LAUNCH_SCRIPT}" "$@"
    else
        printf "\n%s\n" "${delimiter}"
        printf "Launching launch.py..."
        printf "\n%s\n" "${delimiter}"
        prepare_tcmalloc
        "${python_cmd}" "${LAUNCH_SCRIPT}" "$@"
    fi

    if [[ ! -f tmp/restart ]]; then
        KEEP_GOING=0
    fi
done
