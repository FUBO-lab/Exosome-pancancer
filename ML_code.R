set_random_seeds <- function(use_loocv, n_samples, n_folds, n_repeats, tune_length) {
  if (use_loocv) {
    seeds <- vector(mode = "list", length = n_samples + 1)
    for (i in 1:n_samples) seeds[[i]] <- sample.int(n = 1000, tune_length)
    seeds[[n_samples + 1]] <- sample.int(1000, 1)
  } else {
    # Repeated CV: 根据折数和重复次数生成
    n <- n_folds * n_repeats
    seeds <- vector(mode = "list", length = n + 1)
    for (i in 1:n) seeds[[i]] <- sample.int(n = 1000, tune_length)
    seeds[[n + 1]] <- sample.int(1000, 1)
  }
  return(seeds)
}

CompareModel <- function(training, testing, method, sig, group1, group2, use_loocv = FALSE, sample_size = "small", 
                         fold = 5, repeats = 3) {

  selected_cols <- colnames(training)[colnames(training) %in% c("group", sig)]
  if (length(selected_cols) < 2) stop("Features or target variable are missing in the dataset.")
  training <- training[, selected_cols]
  testing <- testing[, selected_cols]
  
  generate_grid <- function(sample_size, sig) {
    if (sample_size == "small") {
      return(list(
        nb = expand.grid(fL = c(0, 0.5, 1), usekernel = c(TRUE, FALSE), adjust = c(0.5, 1, 1.5)),
        svmRadialWeights = expand.grid(sigma = c(0.001, 0.01, 0.1), C = c(1, 3, 10), Weight = c(0.1, 0.5, 1, 2, 3, 5, 10)),
        rf = expand.grid(mtry = seq(1, floor(sqrt(length(sig))), by = 1)),
        kknn = expand.grid(kmax = seq(3, 15, by = 2), distance = 2, kernel = c("optimal", "rectangular", "triangular")),
        adaboost = expand.grid(nIter = seq(10, 100, by = 10), method = c("Adaboost.M1")),
        LogitBoost = expand.grid(nIter = seq(10, 100, by = 10)),
        glmnet = expand.grid(alpha = seq(0, 1, by = 0.2), lambda = seq(0.01, 0.5, by = 0.05)),
        svmRadial = expand.grid(C = c(0.1, 1, 10), sigma = c(0.001, 0.01, 0.1)),
        gbm = expand.grid(interaction.depth = c(1, 3), n.trees = c(50, 100), shrinkage = c(0.01, 0.1), n.minobsinnode = c(5, 10)),
        mlp = expand.grid(size = c(1, 3, 5))
      ))
    } else {
      return(list(
        nb = expand.grid(fL = seq(0, 5, by = 0.5), usekernel = c(TRUE, FALSE), adjust = seq(0.5, 2, by = 0.25)),
        svmRadialWeights = expand.grid(sigma = c(0.0001, 0.001, 0.01, 0.1, 1), C = c(0.1, 1, 10, 100), Weight = c(0.1, 1, 5, 10, 20)),
        rf = expand.grid(mtry = seq(1, floor(sqrt(length(sig))), by = 1)),
        kknn = expand.grid(kmax = seq(3, 50, by = 2), distance = c(1, 2), kernel = c("optimal", "rectangular", "triangular", "epanechnikov")),
        adaboost = expand.grid(nIter = seq(50, 500, by = 50), method = c("Adaboost.M1", "Real adaboost")),
        LogitBoost = expand.grid(nIter = seq(50, 500, by = 50)),
        glmnet = expand.grid(alpha = seq(0, 1, by = 0.1), lambda = seq(0.001, 1, by = 0.01)),
        gbm = expand.grid(interaction.depth = c(1, 3, 5, 7, 9), n.trees = c(50, 100, 200, 300, 500), shrinkage = c(0.01, 0.1, 0.2, 0.3), n.minobsinnode = c(5, 10, 15, 20)),
        mlp = expand.grid(size = c(1, 5, 10, 20), decay = c(0.0001, 0.001, 0.01))
      ))
    }
  }
  Grid <- generate_grid(sample_size, sig)
  
  TuneLength <- sapply(Grid, nrow)
  
  n_samples <- nrow(training)
  seeds <- set_random_seeds(use_loocv, n_samples, fold, repeats, max(TuneLength))
  
  ctrl <- if (use_loocv) {
    trainControl(method = "LOOCV", summaryFunction = twoClassSummary, classProbs = TRUE, seeds = seeds)
  } else {
    trainControl(method = "repeatedcv", number = fold, repeats = repeats, summaryFunction = twoClassSummary, classProbs = TRUE, seeds = seeds)
  }

  ls_model <- lapply(method, function(m) {
    train(group ~ ., data = training, method = m, metric = "ROC", trControl = ctrl, tuneGrid = Grid[[m]])
  })
  
  ls_importance <- lapply(ls_model, function(model) {
    if ("predictor" %in% class(model)) {
      return(NULL)  # Handle other model types accordingly
    } else {
      importance <- varImp(model)
      return(importance)
    }
  })

  evaluate_model <- function(model, dataset, group1, group2) {
    prob <- predict(model, dataset[, -1], type = "prob")
    pre <- predict(model, dataset[, -1])
    test_set <- cbind(as.data.frame(prob),obs = dataset$group, pred = pre)
    rs <- twoClassSummary(test_set, lev = levels(test_set$obs)) %>% as.data.frame() %>% t() %>% as.data.frame()
    rownames(rs) <- model$method

    metrics <- data.frame()
    for (id in c("Accuracy", "Recall", "Precision", "F1", "Kappa")) {
      if (id == "Accuracy") {
        metrics[1,id] <- caret::postResample(pred = test_set$pred, obs = test_set$obs)[["Accuracy"]]
      } else if (id == "Kappa") {
        metrics[1,id] <- caret::confusionMatrix(reference = test_set$obs, data = test_set$pred)$overall["Kappa"]
      } else {
        metrics[1,id] <- caret::confusionMatrix(
          reference = test_set$obs,
          data = test_set$pred,
          positive = group1
        )[["byClass"]][[id]]
      }
    }
    
    rs <- cbind(rs, metrics)
    return(rs)
  }
  
  compute_auc_ci <- function(model, dataset, group1, group2, n_boot = 2000, conf_level = 0.95) {

    prob <- predict(model, dataset[, -1], type = "prob")[, group1]  
    labels <- dataset$group  
    

    roc_obj <- roc(labels, prob, levels = c(group2, group1), direction = "<")
    
    ci <- ci.auc(roc_obj, method = "bootstrap", boot.n = n_boot, conf.level = conf_level)

    return(data.frame(AUC = roc_obj$auc, Lower_2.5 = ci[1], Upper_97.5 = ci[3], model = model[["method"]]))
  }
  
  auc_ci_train <- lapply(ls_model, function(model) compute_auc_ci(model, training, group1, group2))
  auc_ci_test <- lapply(ls_model, function(model) compute_auc_ci(model, testing, group1, group2))

  auc_ci_train_df <- do.call(rbind, auc_ci_train)
  auc_ci_train_df$type <- "training"
  
  auc_ci_test_df <- do.call(rbind, auc_ci_test)
  auc_ci_test_df$type <- "validation"
  
  AUC_CI <- rbind(auc_ci_train_df, auc_ci_test_df)
  
  auc_train <- c()
  for (id in 1:length(ls_model)) {
    aa <- evaluate_model(ls_model[[id]], training, group1, group2)
    auc_train <- rbind(auc_train,aa)
  }
  auc_train$type <- "training"
  
  auc_test <- c()
  for (id in 1:length(ls_model)) {
    aa <- evaluate_model(ls_model[[id]], testing, group1, group2)
    auc_test <- rbind(auc_test,aa)
  }
  auc_test$type <- "validation"
  AUC <- rbind(auc_train, auc_test)

  res <- list(model = ls_model, ROC = AUC, AUC_ci = AUC_CI,importance = ls_importance, seeds = seeds)
  return(res)
}
