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

mkdir -p outputs/entry
cp setup.sh outputs/entry/setup.sh
cp next.sh outputs/entry/next.sh

cp AUTHORS.txt outputs/entry/
cp LICENSE.txt outputs/entry/
cp dependencies.txt outputs/entry/

cp -R biosppy /outputs/entry/
cp -R features /outputs/entry/
cp -R loading /outputs/entry/
cp -R models /outputs/entry/
cp -R preprocessing /outputs/entry/
cp -R utils /outputs/entry/

cp model.pkl outputs/entry || true
cp weights.h5 outputs/entry || true

mkdir outputs/entry/packages
cp -R packages_common/* outputs/entry/packages
cp -R packages_nn/* outputs/entry/packages

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
