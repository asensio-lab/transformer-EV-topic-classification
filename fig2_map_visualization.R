library(ggplot2)      # For plotting
library(tmap)         # For creating tmap
library(tmaptools)    # For reading and processing spatial data related to tmap
library(sf)           # For reading, writing and working with spatial objects
library(readr)
library(cartography)


# Code adapted from http://zevross.com/blog/2018/10/02/creating-beautiful-demographic-maps-in-r-with-the-tidycensus-and-tmap-packages/


# Download "NCHSURCodes2013.csv"from https://www.cdc.gov/nchs/data_access/urban_rural.html
class_df = read_csv("NCHSURCodes2013.csv")

# Choose "FIPS code" and "2013 code" columns
class_df = class_df[c(1,7)]

# Change "FIPS code" column name to "FIPS"
colnames(class_df) = c("FIPS", "Class")

# Convert FIPS code to 5 digit format
class_df$FIPS = as.numeric(class_df$FIPS)
FIPS = sprintf("%05d", class_df$FIPS)
class_df$FIPS = FIPS

# Subset MSAs and mSAs
msa_df = class_df[class_df$Class<5,]
micro_df = class_df[class_df$Class==5,]


# Read in processed data
county_props = read_csv("fig2_data.csv")


# Create subsets based on census region
props_MW = county_props[county_props$census_region=="Midwest",]
props_NE = county_props[county_props$census_region=="Northeast",]
props_S  = county_props[county_props$census_region=="South",]
props_W  = county_props[county_props$census_region=="West",]

# Get quantiles of Availability discussion frequency on 
# 45%, 70%, and 90% quantiles per census region and assign
# "Rarely", "Sometimes", "A Moderate Amount", and "A Great Deal".
quant_MW = quantile(props_MW$Availability, c(0.45, 0.70, 0.90), na.rm = TRUE)
quant_MW

i= "Availability"
props_MW[i] = ifelse(props_MW[i] < quant_MW[1],"Rarely",
                         ifelse(props_MW[i] < quant_MW[2],"Sometimes",
                                ifelse(props_MW[i] < quant_MW[3],"A Moderate Amount","A Great Deal")))

quant_NE = quantile(props_NE$Availability, c(0.45, 0.70, 0.90), na.rm = TRUE)
quant_NE

props_NE[i] = ifelse(props_NE[i] < quant_NE[1],"Rarely",
                     ifelse(props_NE[i] < quant_NE[2],"Sometimes",
                            ifelse(props_NE[i] < quant_NE[3],"A Moderate Amount","A Great Deal")))

quant_S = quantile(props_S$Availability, c(0.45, 0.70, 0.90), na.rm = TRUE)
quant_S

props_S[i] = ifelse(props_S[i] < quant_S[1],"Rarely",
                     ifelse(props_S[i] < quant_S[2],"Sometimes",
                            ifelse(props_S[i] < quant_S[3],"A Moderate Amount","A Great Deal")))


quant_W = quantile(props_W$Availability, c(0.45, 0.70, 0.90), na.rm = TRUE)
quant_W

props_W[i] = ifelse(props_W[i] < quant_W[1],"Rarely",
                     ifelse(props_W[i] < quant_W[2],"Sometimes",
                            ifelse(props_W[i] < quant_W[3],"A Moderate Amount","A Great Deal")))

# Combine the discussion level data to one dataframe
df_avail = rbind(props_MW,props_NE,props_S,props_W)

# Download the data from 
# https://zevross-spatial.github.io/zevross-blogposts/tmap_tidycensus/acs_2012_2016_county_us_B27001.zip
# and extract the zip file to use "acs_2012_2016_county_us_B27001.shp".

# Read in base shape file for map visualization
shp <- st_read("acs_2012_2016_county_us_B27001.shp",
               stringsAsFactors = FALSE)

# Change "GEOID" column to "FIPS" 
colnames(shp)[1]="FIPS"

# Merge data to shapefile data
shp = merge(shp, df_avail[c("FIPS","Availability", "census_region")], by= "FIPS", all = TRUE)

# Assign "Metro" for MSAs and "Micro" for mSAs, and NA for non-core in new column "MSA"
shp["MSA"] = ifelse(is.element(shp$FIPS, msa_df$FIPS),"Metro",
                    ifelse(is.element(shp$FIPS,micro_df$FIPS),"Micro",NA))

# Assign "No Reviews" for NA values in "Availability" column
shp[is.na(shp$Availability),"Availability"] = "No Reviews"

# Assign "Not MSA" (= non-core) for NA values in "MSA" column
shp[is.na(shp$MSA),"MSA"] = "Not MSA"

# Creat a copy of shp
shp_msa=shp

# Assign NA value to "Availability" column, where "MSA" column has value "Not MSA".
shp_msa[shp_msa$MSA=="Not MSA","Availability"]<-NA

# Assign "Availability" levels
cuts <- c("Rarely","Sometimes","A Moderate Amount",
          "A Great Deal")

# Subset MSA only and mSA only data for later use.
metro_only = subset(shp_msa, MSA=="Metro")
micro_only = subset(shp_msa, MSA=="Micro")

# Assign colors for each levels of "Availability" values
# Use color brewer package generate color palettes
mycols = rev(c("#6BAED6","#9ECAE1","grey80",  "#2171B5","#08306B"))



# Create base map.
mymap = tm_shape(shp_msa) + 
  
  tm_polygons("Availability",
          breaks=cuts,
          palette=mycols,
          lwd = 0.3,
              border.col = "grey70",
              border.alpha = 0.25,
              textNA="Not MSA",
              colorNA="white")+
  tm_layout(inner.margins = c(0.06, 0.1,0.1,0.08),
            legend.stack="horizontal",
            frame=FALSE)+
  tm_legend(legend.position=c("left","bottom"))


# Download US states and census regions boundary shape files from 
# https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html

# Read in US states border shape file
states <- st_read("cb_2018_us_state_20m.shp",
               stringsAsFactors = FALSE)

# Add state border layer to the baseline map.
states_border = mymap+ tm_shape(states) + tm_borders(col="black", lwd = 0.7) 

# Read in census region border shape file
regions <- st_read("cb_2018_us_region_20m.shp")

# Add census region border layer to the map
final =states_border + tm_shape(regions) + tm_borders(col = "black", lwd = 2, alpha = 0.5)

# View final map
final

# Save map
tmap_save(final, "figure_2.pdf")

# Create hatched layer for micropolitan areas.
hatchedLayer(micro_only,"right2left", density = 10, lwd = 0.3, col = "grey50")


