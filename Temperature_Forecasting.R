library(tidymodels)
library(modeltime)
library(tidyverse)
library(lubridate)
library(timetk)
library(inspectdf)
library(data.table)
library(dplyr)
library(h2o)
library(rsample)
library(forecast)
library(highcharter)
library(earth)

setwd("C:/Users/99470/Downloads")
df <- fread("C:/Users/99470/Downloads/daily-minimum-temperatures-in-me (1).csv")

df %>% glimpse()

df %>% inspect_na()

names(df) <- names(df) %>% gsub(" ","_",.)

#changing data types
df$Daily_minimum_temperatures <-  parse_number(df$Daily_minimum_temperatures)
df$Date <- df$Date %>% as.Date("%m/%d/%Y")

#visualize
df %>% plot_time_series(Date, Daily_minimum_temperatures,
                        # .color_var=lubridate::week(Date),
                        # .color_lab="Month",
                        .interactive = T,
                        .plotly_slider = T,
                        .smooth=F)

#visualize seasonality
df %>%  plot_seasonal_diagnostics(Date, Daily_minimum_temperatures, .interactive = T)


#date-i hisselere ayririq, lazimsizlari cixiriq
df <- df %>% tk_augment_timeseries_signature(Date) %>% select(Daily_minimum_temperatures,everything())
df <- df %>%
  select(-contains("hour"),
         -contains("day"),
         -contains("week"),
         -minute,-second,-am.pm) %>% 
  mutate_if(is.ordered, as.character) %>% 
  mutate_if(is.character,as_factor)


# Modeling ----------------------------------
h2o.init()
h2o_data <- df %>% as.h2o()

#test/train split
train <- df %>% filter(year < 1988) %>% as.h2o()
test <- df %>% filter(year >= 1988) %>% as.h2o()

target <- df[,1] %>% names() 
features <- df[,-1] %>% names()

model <- h2o.automl(
  x = features, y = target, 
  training_frame = train, 
  validation_frame = test,
  leaderboard_frame = test,
  stopping_metric = "RMSE",
  seed = 123, nfolds = 10,
  exclude_algos = c("GLM","DRF","GBM","XGBoost"),
  max_runtime_secs = 360)

model@leaderboard %>% as.data.frame()
model <- model@leader

#predictions
pred <- model %>% h2o.predict(test) 
pred$predict

model %>% 
  h2o.rmse(train = T,
           valid = T,
           xval = T)

error_tbl <- df %>% 
  filter(lubridate::year(Date) >= 1988) %>% 
  add_column(pred = pred %>% as_tibble() %>% pull(predict)) %>%
  rename(actual = Daily_minimum_temperatures) %>% 
  select(Date,actual,pred)

highchart() %>% 
  hc_xAxis(categories = error_tbl$Date) %>% 
  hc_add_series(data=error_tbl$actual, type='line', color='red', name='Actual') %>% 
  hc_add_series(data=error_tbl$pred, type='line', color='green', name='Predicted') %>% 
  hc_title(text='Predict')


# Forecasting ----
new_data <- seq(as.Date("1991/01/01"), as.Date("1991/12/31"), "days") %>%
  as_tibble() %>% 
  add_column(Daily_minimum_temperatures=0) %>% 
  rename(Date=value) %>% 
  tk_augment_timeseries_signature() %>%
  select(-contains("hour"),
         -contains("day"),
         -contains("week"),
         -minute,-second,-am.pm) %>% 
  mutate_if(is.ordered, as.character) %>% 
  mutate_if(is.character,as_factor)

new_h2o <- new_data %>% as.h2o()

new_predictions <- model %>% 
  h2o.predict(new_h2o) %>% 
  as_tibble() %>%
  add_column(Date=new_data$Date) %>% 
  select(Date,predict) %>% 
  rename(Daily_minimum_temperatures=predict)

df %>% 
  bind_rows(new_predictions) %>% 
  mutate(colors=c(rep('Actual',3650),rep('Predicted',365))) %>% 
  hchart("line", hcaes(Date, Daily_minimum_temperatures, group = colors)) %>% 
  hc_title(text='Forecast') %>% 
  hc_colors(colors = c('red','orange'))


# Model evaluation-------------------
test_set <- test %>% as.data.frame()
residuals = test_set$Daily_minimum_temperatures - pred$predict
residuals
range(residuals)
# Calculate RMSE (Root Mean Square Error) ----
RMSE = sqrt(mean(residuals^2))
RMSE
# Calculate Adjusted R2 (R Squared) ----
y_test_mean = mean(test_set$Daily_minimum_temperatures)
tss = sum((test_set$Daily_minimum_temperatures - y_test_mean)^2) #total sum of squares
rss = sum(residuals^2) #residual sum of squares
R2 = 1 - (rss/tss); R2

n <- test_set %>% nrow() #sample size
k <- features %>% length() #number of independent variables
Adjusted_R2 = 1-(1-R2)*((n-1)/(n-k-1))
Adjusted_R2

tibble(RMSE = round(RMSE,1),
       R2, Adjusted_R2)

#Time Series Models__________________________________

# Split Data 
splits <- initial_time_split(df, prop = 0.8)

#modelde bir dene sorussa da men oyrenmek meqsedli bir necesini qurdum

# Model 1: auto_arima ----
model_fit_arima_no_boost <- arima_reg() %>%
  set_engine(engine = "auto_arima") %>%
  fit(Daily_minimum_temperatures ~ Date, data = training(splits))


# Model 2: arima_boost ----
model_fit_arima_boosted <- arima_boost(
  min_n = 2,
  learn_rate = 0.015
) %>%
  set_engine(engine = "auto_arima_xgboost") %>%
  fit(Daily_minimum_temperatures ~ Date,
      data = training(splits))


# Model 3: ets ----
model_fit_ets <- exp_smoothing() %>%
  set_engine(engine = "ets") %>%
  fit(Daily_minimum_temperatures ~ Date, data = training(splits))


# Model 4: prophet ----
model_fit_prophet <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(Daily_minimum_temperatures ~ Date, data = training(splits))


# Model 5: lm ----
model_fit_lm <- linear_reg() %>%
  set_engine("lm") %>%
  fit(Daily_minimum_temperatures ~ as.numeric(Date),
      data = training(splits))

# Model 6: earth ----
model_spec_mars <- mars(mode = "regression") %>%
  set_engine("earth") 

recipe_spec <- recipe(Daily_minimum_temperatures ~ Date, data = training(splits)) %>%
  step_date(Date, features = "month", ordinal = FALSE) %>%
  step_mutate(date_num = as.numeric(Date)) %>%
  step_normalize(date_num) %>%
  step_rm(Date)

wflw_fit_mars <- workflow() %>%
  add_recipe(recipe_spec) %>%
  add_model(model_spec_mars) %>%
  fit(training(splits))


#Add fitted models to a Model Table.----

models_tbl <- modeltime_table(
  model_fit_arima_no_boost,
  model_fit_arima_boosted,
  model_fit_ets,
  model_fit_prophet,
  model_fit_lm,
  wflw_fit_mars
)

models_tbl




#Calibrate the model to a testing set.----
calibration_tbl <- models_tbl %>%
  modeltime_calibrate(new_data = testing(splits))

calibration_tbl

#Testing Set Forecast & Accuracy Evaluation_________

#Visualizing the Forecast Test
calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = df
  ) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25,
    .interactive      = T
  )


#Accuracy Metrics----
calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = T
  )
#we choose Prophet as it has lowest RMSE

#Refit to Full Dataset & Forecast Forward----

#note:normalda bu asagida bir model secmek lazim deyil, case study ucun etdim
calibration_tbl <- model_fit_prophet %>%
  modeltime_calibrate(new_data = testing(splits))


calibration_tbl %>%
  modeltime_forecast(h = "1 years", actual_data = df) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25,
    .interactive      = T
  )