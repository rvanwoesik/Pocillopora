##############################################################################

############# Deep Learning


#########################################################################

# Intraspecific diversity using machine learning 
# van Woesik, August_September 2024

# Deep learning using multilayer feed-forward artificial neural network


#########################################################################

#                Set up h2o

#######################################################################
#install.packages("h2o")
library(h2o)
library(tidyverse)
h2o.init(nthreads = -1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "8G")  #max mem size is the maximum memory to allocate to H2O

#Set directory
setwd("C://RobsR/Maria_Carr")

#data<-read.csv("nucleotide_diversity.csv", header=TRUE, sep=",") 
data<-read.csv("20240910_nucleotide_diversity.csv", header=TRUE, sep=",")
#data<-read.csv("Cleaned.csv", header=TRUE, sep=",")

head(data)



######################################################################

#                    Preparing data for h2o

####################################################################

#Classifying Site, Ecoregions, Ocean_regions, and Oceans as factors
data$Ecoregion=as.factor(data$Ecoregion)
data$Ocean_region=as.factor(data$Ocean_region)
data$Ocean=as.factor(data$Ocean)

data1=as.h2o(data)

head(data1)
dim(data1)
h2o.anyFactor(data1)

y<- "Nucleotide_diversity"
#y="EISG_site_nucdiv"
x <- c("Longitude", "Ecoregion", "Reef_density", "Mean_SST", "Latitude", "Number_samples", "Ocean_region", "Ocean")

###################################################################

# Partition the data into training, validation and test sets

#####################################################################

splits <- h2o.splitFrame(data = data1, 
                         ratios = c(0.7, 0.15),  #partition data into 70%, 15%, 15% chunks
                         seed = 1)  #setting a seed will guarantee reproducibility
train_h2o <- splits[[1]]
valid_h2o<- splits[[2]]
test_h2o <- splits[[3]]

########################################################

#                 Deep Learning
# H2O's Deep Learning algorithm is a multilayer feed-forward artificial neural network.  

#######################################################

dl_fit1 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train_h2o,
                            model_id = "dl_fit1",
                            seed = 1)
dl_fit1

##############################################################

# Train a DL with new architecture and more epochs (epochs are how many times the model goes through the data).

##############################################################

dl_fit2 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train_h2o,
                            model_id = "dl_fit2",
                            #validation_frame = valid,  #only used if stopping_rounds > 0
                            epochs = 20,
                            hidden= c(10,10),
                            stopping_rounds = 0,  # disable early stopping
                            seed = 1)
dl_fit2

##########################################################

# Train a DL with early stopping

###########################################################
dl_fit3 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train_h2o,
                            model_id = "dl_fit3",
                            #validation_frame = valid,  #in DL, early stopping is on by default
                            epochs = 20,
                            hidden = c(10,10),
                            score_interval = 1,           #used for early stopping
                            stopping_rounds = 3,          #used for early stopping
                            stopping_metric = "deviance",      #used for early stopping
                            stopping_tolerance = 0.0005,  #used for early stopping
                            seed = 1)

dl_fit3

###############################################################

#                Compare models

###############################################################
dl_fit1
dl_fit2
dl_fit3

# Let's compare the performance of the three DL models
dl_perf1 <- h2o.performance(model = dl_fit1,
                            newdata = test_h2o)

dl_perf2 <- h2o.performance(model = dl_fit2,
                            newdata = test_h2o)

dl_perf3 <- h2o.performance(model = dl_fit3,
                            newdata = test_h2o)

#######################################################

#    Plot variable importance 

######################################################

h2o.varimp_plot(dl_fit1, num_of_features = 8)
h2o.varimp_plot(dl_fit2, num_of_features = 8)
h2o.varimp_plot(dl_fit3, num_of_features = 8)

########################################################

#            Plot partial dependency plots

########################################################

#png(file="C:/RobsR/Maria_Carr/DL1_Exposure.png",width=1800, height=800)
par(mfrow=c(2,4))
h2o.partialPlot(object = dl_fit1, newdata = data1,
                cols = c("Longitude", "Ecoregion", "Reef_density", "Mean_SST", "Latitude", "Number_samples", "Ocean_region", "Ocean"))


h2o.partialPlot(object = dl_fit2, data = data1,
                cols = c("Longitude", "Ecoregion", "Reef_density", "Mean_SST", "Latitude", "Number_samples", "Ocean_region", "Ocean"))

h2o.partialPlot(object = dl_fit3, data = data1,
                cols = c("Longitude", "Ecoregion", "Reef_density", "Mean_SST", "Latitude", "Number_samples", "Ocean_region", "Ocean"))


########################################################

# Making the plot as a png

#########################################################

png(file="C:/RobsR/h2o/BLEACHING/DL3_revised_Partial.png",width=1800, height=800)
par(mfrow=c(2,4))
h2o.partialPlot(object = dl_fit2, data = data1, 
                cols = c("Longitude", "Ecoregion", "Reef_density", "Mean_SST", "Latitude", "Number_samples", "Ocean_region", "Ocean"))
                dev.off()

#######################################################

#Grid Search Deep Learning for different hidden layers and l1, regularization options 

########################################################

hidden_opt <- list(c(10,10),c(32,32), c(32,16,8), c(100))
l1_opt <- c(0.005, 1e-4,1e-3)
hyper_params <- list(hidden = hidden_opt, l1 = l1_opt)

model_grid <- h2o.grid("deeplearning",
                       grid_id = "mygrid",
                       hyper_params = hyper_params,
                       x = x,
                       y = y,
                       distribution = "AUTO",
                       training_frame = train_h2o,
                       validation_frame = test_h2o,
                       score_interval = 2,
                       epochs = 20,
                       stopping_rounds = 3,
                       stopping_tolerance = 0.005,
                       stopping_metric = "deviance")
model_grid

#Using this information on the hidden option, apply to new model
dl_fit4 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train_h2o,
                            model_id = "dl_fit4",
                            l1=0.00042,
                            #validation_frame = valid,  #in DL, early stopping is on by default
                            epochs = 20,
                            hidden = c(32,32),
                            score_interval = 1,           #used for early stopping
                            stopping_rounds = 3,          #used for early stopping
                            stopping_metric = "deviance",      #used for early stopping
                            stopping_tolerance = 0.005,  #used for early stopping
                            seed = 1)

dl_fit4

par(mfrow=c(2,4))
h2o.partialPlot(object = dl_fit4, newdata = data1, 
                cols = c("Longitude", "Ecoregion", "Reef_density", "Mean_SST", "Latitude", "Number_samples", "Ocean_region", "Ocean"))

dev.off()


#################################################################

#          Random grid
#Test a series of hidden layer options and regularization (l1) options

#################################################################

#hidden_opt = lapply(1:100, function(x)10+ sample(50,sample(4), replace=TRUE))
hidden_opt <- list(c(10,10),c(32,32), c(32,16,8), c(100))
#l1_opt <- c(0.005, 1e-4,1e-3)
l1_opt = seq(1e-6,1e-3,1e-6)
hyper_params <- list(hidden = hidden_opt, l1 = l1_opt)
search_criteria = list(strategy = "RandomDiscrete",
                       max_models = 15, max_runtime_secs = 100,
                       seed=123456)
search_criteria

model_grid <- h2o.grid("deeplearning",
                       grid_id = "mygrid",
                       hyper_params = hyper_params,
                       search_criteria = search_criteria,
                       x = x,
                       y = y,
                       distribution = "AUTO",
                       training_frame = train_h2o,
                       validation_frame = test_h2o,
                       score_interval = 2,
                       epochs = 2000,
                       stopping_rounds = 3,
                       stopping_tolerance = 0.005,
                       stopping_metric = "deviance")                     

model_grid


#      hidden      l1    model_ids          residual_deviance

#Hidden layers 32,32 with l1 = 0.00042 regularization to prevent overfitting

#Best model

 
#load(h2o.partialPlot2)
par(mfrow=c(2,4))
h2o.partialPlot(object = dl_fit5, newdata = data1, 
                cols = c("Longitude", "Ecoregion", "Reef_density", "Mean_SST", "Latitude", "Number_samples", "Ocean_region", "Ocean"))

dev.off()


###################################################################

#           Saving the best model

####################################################################

#h2o.saveModel(object = dl_fit5, path = getwd(), force = TRUE)
#print(model_path)

# load the model
#dl_fit5 <- h2o.loadModel("C:\\RobsR\\Maria_Carr\\dl_fit5")

# download the model built above to your local machine
#dl_fit5 <- h2o.download_model(dl_fit5, path = "C:\\RobsR\\Maria_Carr\\dl_fit5")

# upload the model that you just downloaded above
#dl_fit5 <- h2o.upload_model(dl_fit5)

dl_fit5=save(dl_fit5, file="Pocillopora")


####################################################################

#                 Check feature importance

###################################################################

library(iml)
library(maxnet)

#X1=data %>% select(5,6,7,8,9,10,11)
X1=data[,4:11]
#X1=data[,5:12]
head(X1)

predictor<- Predictor$new(dl_fit5,data=X1, y=data$Nucleotide_diversity)
#predictor<- Predictor$new(dl_fit5,data=X1, y=data$EISG_site_nucdiv)
imp<-FeatureImp$new(predictor, loss="mae")

library(ggplot2)
myplot=plot(imp, cex.names=1.5, cex.lab=1.5, cex.axis=1.3)
myplot + theme_bw() +
theme(text = element_text(size=20)) +
xlab("Feature importance (Mean absolute error)")  

imp$results
#feature importance.05 importance importance.95 permutation.error
#1       Mean_SST     1.2280612   1.253075      1.391842       0.001632315
#2   Reef_density     1.1328101   1.251506      1.279701       0.001630270
#3   Ocean_region     1.1222468   1.211759      1.301052       0.001578494
#4          Ocean     1.0240683   1.034139      1.072711       0.001347117
#5 Number_samples     1.0026940   1.013881      1.054249       0.001320729
#6      Longitude     0.9652578   1.011686      1.037993       0.001317870
#7      Ecoregion     0.9702883   1.010126      1.060905       0.001315838
#8       Latitude     0.9799662   0.995056      1.046511       0.001296207


#h2o.varimp_plot(dl_fit5, num_of_features = 8) #Doesn't work for categorical data



#########################################################

#            Andy Walker's code for partial dependency plots

#########################################################
#Figure 4

library(scales)
library(ggplot2)
library(ggpubr)

# You must have a model and h2o dataset prior to this code

partialplot = h2o.partialPlot(dl_fit5, data1, plot = F)

Modify_Partial_Plot = function(partialplot, color, varname, x.angle, vjust) {
  
  for (i in 1:length(partialplot)) {
    if (names(partialplot[[i]])[1] == varname) {
      val = i
      break
    }
  }
  layer = partialplot[[i]]
  
  
    #for discrete variables
  if (data.class(layer[,1]) == "character" | data.class(layer[,1]) == "factor") {
    #ord = layer[order(layer$mean_response, decreasing = F), 1] # Ordering the levels in a increasing manner
    #layer[,1] = factor(layer[,1], levels = ord)
    
    # Plotting
    p = ggplot(layer, aes(x = layer[,1], y = mean_response)) + 
      geom_point(color = color, cex = 3) +
      geom_errorbar(aes(ymin=mean_response-stddev_response , ymax=mean_response+stddev_response), 
                    color = color, # You can change the color manually here
                    width=.2,
                    position=position_dodge(.9)) + 
      #coord_fixed(ratio = 0.12) +
      xlab(varname) +
      ylab("Genetic diversity") + # Change the y axis label here
      theme(panel.background = element_blank(), # Removing background
            panel.grid.major = element_blank(), # Removing grid
            panel.grid.minor = element_blank(), # Removing grid
            panel.border = element_rect(colour = "black", fill=NA, size=0.5), # Creating border
            axis.title = element_text(size = rel(1.0)), # The rel() function makes the axis text and axis title the same size
            axis.text.y = element_text(color = "black",
                                       size = rel(1.0)),
            axis.text.x = element_text(color = "black",
                                       size = rel(1.0), angle = x.angle, vjust = vjust),
            # Increasing the spacing between the axis title and the axis text
            axis.title.x = element_text(margin = margin(t = 10)),
            axis.title.y = element_text(margin = margin(r = 10)),
            # Adding margin to the edges of the plot to prevent the axis texts from getting cutoff
            plot.margin = margin(t = 10, r = 25, b = 10, l = 10)
      )
  }
  
  # If continuous
  if (data.class(layer[,1]) == "numeric") {
    
    
    
    # Plotting
    p = ggplot(layer, aes(x=layer[,1], y=mean_response)) + 
      geom_line(color = color) +  # Line for mean response
      geom_ribbon(aes(ymin = mean_response - stddev_response, ymax = mean_response + stddev_response), 
                  alpha = 0.2, fill = color) +  # Shaded region for stddev
      #coord_fixed(ratio = 0.2*((max(layer[,1]) - min(layer[,1]))/(max(layer$mean_response) - min(layer$mean_response)))) +
      xlab(varname) +
      ylab("Genetic diversity") + # Change the y axis label here
      scale_x_continuous(n.breaks = 5) +
      theme(panel.background = element_blank(), # Removing background
            panel.grid.major = element_blank(), # Removing grid
            panel.grid.minor = element_blank(), # Removing grid
            panel.border = element_rect(colour = "black", fill=NA, size=0.5), # Creating border
            axis.title = element_text(size = rel(1.0)), # The rel() function makes the axis text and axis title the same size
            axis.text.y = element_text(color = "black",
                                     size = rel(1.0)),
            axis.text.x = element_text(color = "black",
                                       size = rel(1.0), angle = x.angle, vjust = vjust),
            # Increasing the spacing between the axis title and the axis text
            axis.title.x = element_text(margin = margin(t = 10)),
            axis.title.y = element_text(margin = margin(r = 10)),
            # Adding margin to the edges of the plot to prevent the axis texts from getting cutoff
            plot.margin = margin(t = 10, r = 25, b = 10, l = 10)
      )
    
  }
  
  return(p)
}

print(x) # varnames to call in

# Example plots
p1 = Modify_Partial_Plot(partialplot = partialplot, color = "blue", varname = "Ocean_region", x.angle = 0, vjust = 0.4)
p2 = Modify_Partial_Plot(partialplot = partialplot, color = "blue", varname = "Ecoregion", x.angle = 45,vjust = 0.4)
p3 = Modify_Partial_Plot(partialplot = partialplot, color = "blue", varname = "Latitude", x.angle = 0, vjust = 0.4)
p4 = Modify_Partial_Plot(partialplot = partialplot, color = "blue", varname = "Longitude", x.angle = 0, vjust = 0.4)
p5 = Modify_Partial_Plot(partialplot = partialplot, color = "blue", varname = "Reef_density", x.angle = 0, vjust = 0.4)
p6 = Modify_Partial_Plot(partialplot = partialplot, color = "blue", varname = "Ocean", x.angle = 0, vjust = 0.4)
p7 = Modify_Partial_Plot(partialplot = partialplot, color = "blue", varname = "Number_samples", x.angle = 0, vjust = 0.4)
p8 = Modify_Partial_Plot(partialplot = partialplot, color = "blue", varname = "Mean_SST", x.angle = 0, vjust = 0.4)

#p1;p2;p3;p4;p5;p6;p7;p8

# You will want to save each plot to a seperate object, and then use ggarrange

# Arranging multiple ggplots in one layout
ggarrange(p1, p2, p3, p4, p5, p6, p7,p8,# specify plots here
          nrow = 2, # number of rows
          ncol = 4) # number of columns
 # you can also use hjust, vjust, widths, and heights to modify the display

#Final arrangement

ggarrange(p6, p1, p2, p3, p4, p5, p8,p7,# specify plots here
          nrow = 2, # number of rows
          ncol = 4) # number of columns



#Save files
#png(file="C:/RobsR/Maria_Carr/Figure 4 revised.png",width=600, height=350)

######################################################################################################


######################################################################################################

