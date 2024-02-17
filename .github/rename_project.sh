#!/usr/bin/env bash
while getopts ":a:n:u:d:" flag
do
    case "${flag}" in
        a) author=${OPTARG};;
        n) name=${OPTARG};;
        u) urlname=${OPTARG};;
        d) description=${OPTARG};;
    esac
done

echo "Author: $author";
echo "Project Name: $name";
echo "Project URL name: $urlname";
echo "Description: $description";

echo "Renaming project..."

original_author="biqute"
original_name="twpasolver"
original_urlname="https://github.com/biqute/twpasolver"
original_description="Project twpasolver created by biqute"
# for filename in $(find . -name "*.*") 
for filename in $(git ls-files) 
do
    sed -i "s/$original_author/$author/g" $filename
    sed -i "s,$original_urlname,$urlname,g" $filename
    sed -i "s/$original_description/$description/g" $filename
    echo "Renamed $filename"
done

for filename in $(git ls-files) 
do
    sed -i "s/$original_name/$name/g" $filename
done
# sed -i '3s/^.//' .github/workflows/analysis.yml
mv src/$original_name src/$name

# This command runs only once on GHA!
rm -rf .github/template.yml

# Remove template description from readme
sed -i '30,$d' README.md
