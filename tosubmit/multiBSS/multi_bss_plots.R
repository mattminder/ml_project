library(glue)
library(RcppCNPy)
library(ggplot2)
library(reshape2)

# Load 
# MODIFY PATH
mypath = "/Users/myfiles/Documents/EPFL/M_I/MLE/ml_project/tosubmit/"

acc_fwd = as.data.frame(npyLoad(glue(mypath, "multiBSS/SVM_computed_accuracy_fwd.npy")))
excl_feat_fwd = as.data.frame(npyLoad(glue(mypath, "multiBSS/SVM_feature_included_mean.npy")))
max_acc_fwd = as.data.frame(npyLoad(glue(mypath, "multiBSS/SVM_max_accuracy_fwd.npy")))

n_feat = 30

# Get names
train = read.csv(glue(mypath,"all/train.csv"), nrows = 1)
feat_names = names(train)[3:ncol(train)]
names(max_acc_fwd) = feat_names
names(excl_feat_fwd) = feat_names

# Line plots
max_acc_mean = apply(max_acc_fwd, 2, mean)
max_acc_std = apply(max_acc_fwd, 2, sd)
x = 1:30
quartz()
plot(x, max_acc_mean, type = "b",pch=19,xlab = "Number Included Features",
     ylab = "Accuracy")
arrows(x, max_acc_mean-max_acc_std, x, max_acc_mean+max_acc_std, length=0.05, angle=90, code=3)
quartz.save(glue(mypath, "multiBSS/accuracy_plot.pdf"), type = "pdf")

# Heatmap
tomelt = cbind(1:30, excl_feat_fwd)
names(tomelt)[1] = "id"

data_melt = melt(tomelt, id.vars = "id")
g = ggplot(data = data_melt, aes(x = id, y = variable)) +
  geom_tile(aes(fill = value), color = "white", size = .1) +
  scale_fill_gradient(low = "gray92", high = "firebrick3") + 
  theme_grey(base_size = 10) + 
  ggtitle("Forward BSS") + xlab("Number Features in Model") + ylab("Features") +
  theme(axis.ticks = element_blank(), 
        panel.background = element_blank(), 
        plot.title = element_text(size = 18, face="bold"),
        axis.text.x = element_text(angle = 90, hjust = 1)) 
ggsave(glue(mypath, "multiBSS/fwd_heat.pdf"), g)

