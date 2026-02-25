cd ..
git fetch --all
git reset --hard origin/master
chmod +x src/build.sh
cd src
./build.sh