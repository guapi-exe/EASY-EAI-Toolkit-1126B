cd ..
git fetch --all
git reset --hard origin/master
chmod +x src/build.sh
chmod +x src/update.sh
cd src
./build.sh