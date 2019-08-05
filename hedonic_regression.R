# In the following code, I create variables for hedonic regression of home values
# in Portland, Oregon. This is in addition to publicly available data such as square 
# footage, lot size, age, elevation, etc. 

# Derived variable #1 - number of bus stops in neighborhood
# using data from Metro - RLIS
require(sp)
bus_neigh <- over(bus_stops, neighborhoods)

require(dplyr)
neigh_bus_counts <- bus_neigh %>%
  group_by(NAME) %>% # group bus stops by neighborhood name
  count() # count 'em!

house_neigh <- over(spTransform(pdx_houses,neighborhoods@proj4string), 
                    neighborhoods)
neigh_bus_counts <- as.data.frame(neigh_bus_counts)
house_neigh$bus_count <- NA

# match neighborhood of each house with bus stop count 
# in that neighborhood, assign to house
for (k in 1:nrow(house_neigh)) {
  x1 <- house_neigh$NAME[k]
  x2 <- which(neigh_bus_counts$NAME == x1)
  x3 <- neigh_bus_counts$n[x2]
  ifelse(x2 == 0, NA, house_neigh$bus_count[k] <- x3)
}
house_neigh$bus_count[is.na(house_neigh$bus_count)] <- 0

# Derived variable #2 - distance to nearest railroad
require(rgeos)
dist_rail <- rgeos::gDistance(spTransform(pdx_houses, 
                                          railroads@proj4string), railroads, byid=TRUE)
dist_rail <- t(dist_rail) # transpose
NND_rail <- unlist(lapply(1:nrow(dist_rail), 
                          function(i) round(dist_rail[i,which.min(dist_rail[i,])],2) 
)) # return min NN dist by row

# Dummy variable #1 - is the house above 600 feet?
require(raster)
# import DEM
rast <- raster("C:/Users/dmitri4/Desktop/597_lab5/elev_16bit.tif") 
# extract elevation at each house
elev_data <- raster::extract(rast, pdx_houses) 
elev_data <- elev_data * 3.28 # convert meters to feet

house_neigh <- cbind(house_neigh, elev_data)
# create the variable!
house_neigh$is_high <- ifelse(house_neigh$elev_data >= 600, 1, 0) 

# Dummy variable #2 - does the address end with "Court"?
require(stringr)
address <- pdx_houses$SITEADDR
address_vec <- str_detect(address,"CT")
# 56 houses have a "Court" address
house_neigh$on_court <- NA
house_neigh$on_court[address_vec == TRUE] <- 1
house_neigh$on_court[address_vec == FALSE] <- 0
