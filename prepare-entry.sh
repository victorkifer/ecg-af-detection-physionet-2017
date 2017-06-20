#!/bin/bash
#
# file: prepare-entry.sh
#
# This script shows how to run the example code (setup.sh and next.sh)
# over the validation set, in order to produce the list of expected
# answers (answers.txt) which must be submitted as part of your entry.
# This script itself does not need to be included in your entry.

set -e
set -o pipefail

FAILED=YES

function cleanup {
    if [ "$FAILED" = "YES" ]; then
        >&2 echo "An error occurred while preparing entry"
    fi
    echo "Cleaning up"
    rm -R outputs/entry || true
}
trap cleanup EXIT

echo "==== running entry script on validation set ===="

rm -R outputs/entry >/dev/null 2>&1 || true
rm -r outputs/entry.zip >/dev/null 2>&1 || true

cp -R physionet_entry outputs/entry

python3 main.py -m classify -o outputs/entry/answers.txt

find . -type d -name '__pycache__' | xargs rm -rf

cp AUTHORS.txt outputs/entry/
cp LICENSE.txt outputs/entry/

cp -r biosppy outputs/entry/biosppy
cp -r features outputs/entry/features
cp -r loading outputs/entry/loading
cp -r models outputs/entry/models
cp -r preprocessing outputs/entry/preprocessing
cp -r utils outputs/entry/utils
cp *.py outputs/entry

cp model.pkl outputs/entry || true
cp weights.h5 outputs/entry || true

read -p "Is this a dry-run entry(not for score)? " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    touch outputs/entry/DRYRUN
fi

cd outputs/entry
zip -r ../entry.zip ./
cd -

echo "Entry was created successfully"
FAILED=NO
