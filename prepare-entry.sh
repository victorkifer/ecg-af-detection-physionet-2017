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

function cleanup {
  echo "Cleaning up"
  rm -R outputs/entry || true
}
trap cleanup EXIT

for i in "$@"
do
case $i in
    -m=*|--model=*)
    MODEL="${i#*=}"
    shift # past argument=value
    ;;
    --dry-run)
    DRY_RUN=YES
    shift # past argument with no value
    ;;
    *)
            # unknown option
    ;;
esac
done

echo "==== running setup script ===="

./setup.sh

echo "==== running entry script on validation set ===="

rm -R outputs/entry > /dev/null || true
rm -r outputs/entry.zip > dev/null || true
rm -f answers.txt

python3 main_machine_learning.py

mkdir -p outputs/entry
cp setup.sh outputs/entry/
cp next.sh outputs/entry/

cp AUTHORS.txt outputs/entry/
cp LICENSE.txt outputs/entry/

mkdir outputs/entry/utils
cp -R utils/*.py outputs/entry/utils/

mkdir outputs/entry/common
cp -R common/*.py outputs/entry/common/

cp requirements.txt outputs/entry/
cp *.py outputs/entry
cp answers.txt outputs/entry

cp model.pkl outputs/entry || true
cp weights.h5 outputs/entry || true

cp DRYRUN outputs/entry/DRY_RUN

cd outputs/entry
zip -r ../entry.zip ./
cd -

echo "Entry was created successfully"