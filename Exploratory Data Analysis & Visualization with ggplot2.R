rm(list=ls(all=TRUE))
setwd("L:\\AFTER\\Kaggle\\train")
library(readr)
library(gcookbook)
library(dplyr)
library(stringr)
library(ggplot2)
library(tokenizers)
library(tm)
library(xgboost)
library(quanteda)

train = read_tsv("train.tsv")
train = data.frame(train)
test = read_tsv("test.tsv")
train = train[which(is.na(train$item_description)==FALSE),]
head(train)
attach(train)
item_condition_id = as.factor(item_condition_id)
shipping = factor(shipping,labels =c("Buyer","Seller"))
log_price = log(price+1)

#Exploratory Data Analyss

#Histogram / density of price
density.price = ggplot(train, aes(x = price))+geom_density()+theme_bw()+
  ggtitle("Density of Price")
density.log.price = ggplot(train, aes(x = log(price+1)))+geom_density()+theme_bw()+
  ggtitle("Density of log(price+1)")


density.price
density.log.price

#Missing Value
apply(train, 2, function(x) sum(is.na(x)))
# train_id              name item_condition_id     category_name 
# 0                 0                 0              6327 
# brand_name             price          shipping  item_description 
# 632682                 0                 0                 4 

# Delect rows with no description


## Shipping vs. Price
shipping.price=ggplot(data=train,aes(x = log(price+1),fill =factor(shipping)))+
  geom_density(adjust = 2,alpha=.5) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5)) +
  ggtitle('Distribution of price vs shipping')
shipping.price
# Products of free shipping are cheaper.

## Item condition vs. Price
condition.price = ggplot(data = train, aes(x = log(price+1), fill =factor(item_condition_id)))+
  geom_density(alpha = 0.5, adjust = 2)+
  theme_bw()+
  theme(plot.title= element_text(hjust = 0.5)) +
  ggtitle('Distribution of price vs condition')
condition.price
ggplot(train)+
  geom_boxplot(aes(x = factor(item_condition_id),y = log(price+1),fill = item_condition_id))+
  labs("Boxplot of log(price+1) vs. condition")

# Condition id is not a good factor for prediction.

## Item Category
#Category or not 
train["cate_or_not"]= ifelse(is.na(category_name), 0, 1)
ggplot(data = train)+
  geom_density(aes(x = log(price+1), fill = factor(cate_or_not)),adjust = 2,alpha=.5)+
  labs(title = "Categorized or not")
# Being categorized or not doesn't matter for the pricing

uniq_ctgy = unique(category_name)
num_uniq_ctgy = length(uniq_ctgy) #1288
sub_ctgy = data.frame(category_name%>%str_split("/",simplify = TRUE))
colnames(sub_ctgy)=c("C1","C2","C3","C4","C5")
train = data.frame(train, sub_ctgy)

# unique sub_ctgy
num_sub_uniq = sub_ctgy%>%summarise(uniq_c1 = length(unique(sub_ctgy$C1)),
                                    uniq_c2 = length(unique(sub_ctgy$C2)),
                                    uniq_c3 = length(unique(sub_ctgy$C3)),
                                    uniq_c4 = length(unique(sub_ctgy$C4)),
                                    uniq_c5 = length(unique(sub_ctgy$C5)))
num_sub_uniq
# uniq_c1 uniq_c2 uniq_c3 uniq_c4 uniq_c5
#      11     114     871       7       3

# missing value for sub categories
miss_sub_uniq = train%>%summarise(mis_1 = sum(is.na(C1)),
                                  mis_2 = sum(C2 == ""),
                                  mis_3 = sum(C3 ==""),
                                  mis_4 = sum(C4 == ""),
                                  mis_5 = sum(C5 == ""))
miss_sub_uniq
# mis_1 mis_2 mis_3   mis_4   mis_5
#   6327  6327  6327 1478142 1479472
# So I only took C1,C2 and C3 into consideration


count_main_uniq = sub_ctgy%>%group_by(C1)%>%summarise(Main_Category = n())%>%arrange(desc(Main_Category))

#Plot for C1
# Ranking
ggplot(data = count_main_uniq)+
  geom_bar(aes(x = reorder(C1,desc(Main_Category)), y = Main_Category),stat = "identity",fill = "lightblue")+
  labs(x = "Main Category",y = "Count",title = "Main Category Count")+
  theme(axis.text.x=element_text(angle=15, hjust=0.5, size=7))
# Boxplot 
ggplot(data = train)+
  geom_boxplot(aes(x = C1,y = log(price+1),fill = C1))+
  theme(axis.text.x=element_text(angle=15, hjust=0.5, size=7))+
  labs(x = "C1", title = "log(price+1) v.s. C1")
#Plot for C2 (Top 25)
# Ranking
count_c2_uniq = sub_ctgy%>%group_by(C2)%>%summarise(C2_Category = n())%>%arrange(desc(C2_Category))
ggplot(data=count_c2_uniq[1:25,])+
  geom_bar(aes(x = reorder(C2,desc(C2_Category)),y =C2_Category),stat = "identity",fill = "lightblue")+
  labs(x = "C2 Category",y = "Count",title = "Sub_main Category Count")+
  theme(axis.text.x=element_text(angle=25, hjust=0.5, size=7))
#Boxplot
ggplot(data = train)+
  geom_boxplot(aes(x = C2, y = log(price+1), fill = C2))+
  labs(x = "C2 ",title = "log(price+1) v.s. C2")+
  theme(axis.text.x=element_text(angle=90, hjust=0.5, size=7),legend.position="none")

## Products description
# Deal with "no description yet"
item_description = ifelse(item_description=="No description yet",NA,item_description)

# Length of description
train = train %>% mutate(len_of_des = str_length(item_description))

train%>%group_by(len_of_des) %>%
  summarise(mean_log_price = mean(log(price+1))) %>% 
  ggplot(aes(x=len_of_des, y=mean_log_price)) +
  geom_point(size=0.5) +
  geom_smooth(method = "loess", color = "red", size=0.5) +
  ggtitle('Mean Log Price versus Length of Description')
# Boxplot for if there is any description
train["des_or_not"]=ifelse(is.na(item_description), 0, 1)
ggplot(data = train,aes(x =log(price+1), fill = factor(des_or_not)))+
  geom_density(adjust = 2,alpha=.5)+labs(title = "Description or not")
ggplot(data=train)+geom_boxplot(aes(x = factor(train$des_or_not), y = log(price+1)))+
  labs(title = "Description or not - Boxplot")

# Brand name
train["brname_or_not"]= ifelse(is.na(brand_name),0, 1)
ggplot(data= train)+geom_density(aes(x = log(price+1), fill = factor(brname_or_not)), 
                                     adjust = 2,alpha=.5)+labs(title = "Brand name or not")
ggplot(data = train)+geom_boxplot(aes(x = factor(brname_or_not), y = log(price+1), fill = factor(brname_or_not)))+
  labs("Brand name or not")

####################################EDA END################################333

####################################NLP START###################################

# Extract features from text
# Brand Name
# Modify the dataset, especially dealing with NAs
brand_name = ifelse(is.na(brand_name), "undefined", brand_name)
category_name = ifelse(is.na(category_name), "undefined", category_name)
item_description = ifelse(item_description == "No description yet", "undefined",item_description)


# Tokenization
Token_des = tokens(train$item_description, what = "word",
                          remove_numbers = TRUE, remove_punct = TRUE,
                          remove_symbols = TRUE, remove_hyphens = TRUE)
Token_des = tokens_tolower(Token_des)
Token = tokens_select(Token_des, stopwords(language = "en"),
                                 selection = "remove")
Token_des = Token
Token_des = tokens_wordstem(Token_des, language = "english")


Token_name = tokens(train$name, what = "word",
                    remove_numbers = TRUE, remove_punct = TRUE,
                    remove_symbols = TRUE, remove_hyphens = TRUE)
Token_name = tokens_tolower(Token_name)
Token_name = tokens_select(Token_name, stopwords(language = "en"),
                      selection = "remove")
Token_name = tokens_wordstem(Token_name, language = "english")

# Bag of word
BOW_des = dfm(Token_des)
dim(BOW_des) # 1482531  138945
trim_BOW_des = dfm_trim(BOW_des, min_count = 300)

BOW_name = dfm(Token_name)
dim(BOW_name)
trim_BOW_name = dfm_trim(BOW_name, min_count = 300)

#TF-IDF
tfidf_des = dfm_tfidf(trim_BOW_des)
tfidf_name = dfm_tfidf(trim_BOW_name)

training = train[1:1000000,]






