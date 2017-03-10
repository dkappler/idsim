#!/bin/bash

# Author: Daniel Kappler
# Please contact daniel.kappler@gmail.com if bugs or errors are found
# in this script.

# The project dir, since we assume we are checked out in the top level dir.
PROJECT_DIR="${UBASH_DIR}/.."


UBASH_OS="linux"
if [[ $OSTYPE == darwin* ]];then
    UBASH_OS="mac"
fi


ubash::pp() {
    echo -e "$1"
}


ubash::command_exists() {
	command -v "$@" > /dev/null 2>&1
}

ubash::user_confirm() {
    local NOT_FINISHED=true
    while ${NOT_FINISHED} ;do
        ubash::pp "$1 [y/n] default($2) "
        read USER_INPUT;
        if [[ "y" == "${USER_INPUT}" ]];then
            USER_CONFIRM_RESULT="y";
            NOT_FINISHED=false;
        elif [[ "n" == "${USER_INPUT}" ]];then
            USER_CONFIRM_RESULT="n";
            NOT_FINISHED=false;
        elif [[ "" == "${USER_INPUT}" ]];then
            USER_CONFIRM_RESULT="$2";
            NOT_FINISHED=false;
        else
            ubash::pp "# only y, n, and nothing, are possible choices."
            ubash::pp "# default is $2"
        fi
    done
}
