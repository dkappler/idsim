#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd)"
source "${SCRIPT_DIR}/ubash.sh" || exit 1

ubash::pp "Create the virtual environment."
ubash::create_virtualenv
ubash::source_virtualenv

ubash::pp "# Should we upgrade all dependencies?"
ubash::user_confirm ">> Update dependencies?" "n"
if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
    ${VPY_PIP} install --upgrade pip
    ${VPY_PIP} install --upgrade \
            scipy jupyter ipython pyyaml ipdb \
            matplotlib jinja2 autopep8 \
            cython colorama easydict
fi


ubash::user_confirm ">> Install tensorflow GPU?" "n"
if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
    ubash::pp "Please install cuda 8 nvidia!"
    ubash::pp "Please install cudnn 5.1 from nvidia!"
    ubash::pp "Notice, symbolic links for libcudnn.dylib and libcuda.dylib have to be added."
    ${VPY_PIP} install tensorflow-gpu 
fi


ubash::user_confirm ">> Install tensorflow CPU?" "n"
if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
    ${VPY_PIP} install tensorflow
fi

cd ${PROJECT_DIR}
${VPY_BIN} setup.py develop
