
VPY_DIR="${PROJECT_DIR}/vpy"
VPY_BIN="${VPY_DIR}/bin/python"
VPY_PIP="${VPY_DIR}/bin/pip"

ubash::create_virtualenv() {
    if [[ ! -e ${VPY_DIR} ]];then
        ubash::pp "# We setup a virtual environemnt ${VPY_DIR}."
        if ! ubash::command_exists virtualenv;then
            ubash::pp "# We install virtualenv!"
            sudo pip install virtualenv 
        fi
        virtualenv ${VPY_DIR} --clear
        virtualenv ${VPY_DIR} --relocatable
    fi
}

ubash::source_virtualenv() {
    if [[ ! -e ${VPY_DIR} ]];then
        ubash::pp "# No virtual environemnt ${VPY_DIR}."
        exit 1
    fi

    source "${VPY_DIR}/bin/activate"
}

ubash::autopep8() {
    if [[ "$#" -ne 1 ]];then
        ubash::pp "# ubash::autopep8 needs one argument the root directory."
    fi
    if ! ubash::command_exists autopep8; then
        ubash::pp "# We need autopep8."
        ubash::pp "# Please install autopep by calling: "
        ubash::pp "# sudo pip install autopep8;"
    else
        ubash::user_confirm "Run autopep for directory $1 and below?" "n"
        if [[ "y" == "${USER_CONFIRM_RESULT}" ]];then
            set -x
            find $1 -name '*.py' -exec autopep8 --in-place -a '{}' \;
        fi
    fi
}
