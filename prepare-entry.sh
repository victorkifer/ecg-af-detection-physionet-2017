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

echo "==== running setup script ===="

#./setup.sh

echo "==== running entry script on validation set ===="

mkdir -p outputs/entry
cp setup.sh outputs/entry/
cp next.sh outputs/entry/

cp AUTHORS.txt outputs/entry/
cp LICENSE.txt outputs/entry/

cp -R utils outputs/entry/
cp requirements.txt outputs/entry/
cp *.py outputs/entry

cd outputs/entry
zip -r ../entry.zip ./
cd -
rm -R outputs/entry