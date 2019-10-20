mkdir -p ncep
cd ncep
for year in 2012 2013 2014 2015 2016 2017 2018 2019
do
  wget -nv --show-progress -nc https://www.esrl.noaa.gov/psd/thredds/fileServer/Datasets/ncep/uwnd.$year.nc
  wget -nv --show-progress -nc https://www.esrl.noaa.gov/psd/thredds/fileServer/Datasets/ncep/rhum.$year.nc
  wget -nv --show-progress -nc https://www.esrl.noaa.gov/psd/thredds/fileServer/Datasets/ncep/air.$year.nc
done
