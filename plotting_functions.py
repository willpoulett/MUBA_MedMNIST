import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sn
from sklearn.calibration import calibration_curve
import pandas as pd


def plot_history(history, size = (20,10), epochs = 20):
    epochs = [i for i in range(epochs)]
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    fig.set_size_inches(20,10)

    ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
    ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
    ax[0].set_title('Training & Validation Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
    ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
    ax[1].set_title('Testing Validation & Loss')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Training & Validation Loss")
    plt.show()

def muba_curve(type,
               values,
               bins = 20,
               average = True,
               title = None,
               fig_size = (10,8),
               xlabel="Alpha",
               ylabel="N"):
    
    assert type == "error" or type == "boundary"

    fig, ax = plt.subplots(figsize=fig_size)
    
    ax.hist(values,bins=[i/bins for i in range(bins+1)])
    if average:
        ax.axvline(np.mean(values), linestyle = "--", label = f"Mean: {round(np.mean(values),2)}")

    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plot_3_confusion_matricies(mixup_images_df,true_images_A_df,true_images_B_df, labels = ["Normal","Pneumonia"]):

    fig, axs = plt.subplots(1,3, figsize = (15,5))
    cm_muba = confusion_matrix([x for x in mixup_images_df["label"]],[y for y in mixup_images_df["argmax_pred"]])
    cm_test_A = confusion_matrix([x for x in true_images_A_df["label"]],[y for y in true_images_A_df["argmax_pred"]])
    cm_test_B = confusion_matrix([x for x in true_images_B_df["label"]],[y for y in true_images_B_df["argmax_pred"]])

    cm_muba_display = ConfusionMatrixDisplay(confusion_matrix = cm_muba, display_labels=labels)
    cm_test_A_display = ConfusionMatrixDisplay(confusion_matrix = cm_test_A, display_labels=labels)
    cm_test_B_display = ConfusionMatrixDisplay(confusion_matrix = cm_test_B, display_labels=labels)

    cm_muba_display.plot(ax = axs[0])
    cm_test_A_display.plot(ax = axs[1])
    cm_test_B_display.plot(ax = axs[2])

    fig.tight_layout()

    axs[0].set_title("Mixup Data")
    axs[1].set_title("Test A Data")
    axs[2].set_title("Test B Data")
    plt.show()

def plot_calibration(true_images_A_df,true_images_B_df,mixup_images_df,boundary_images_df, positive_class = "Normal"):

    fig, axs = plt.subplots(1,2,figsize = (10,5))

    prob_demented_A, prob_pred_A = calibration_curve(true_images_A_df["label"],true_images_A_df["argmax_pred"],n_bins=10)
    prob_demented_B, prob_pred_B = calibration_curve(true_images_B_df["label"],true_images_B_df["argmax_pred"],n_bins=10)
    prob_demented_mix, prob_pred_mix = calibration_curve(mixup_images_df["label"],mixup_images_df["argmax_pred"],n_bins=10)

    # The proportion of samples whose class is the positive class, in each bin (fraction of positives)
    # The mean predicted probability in each bin.

    alpha_confidence_df = pd.DataFrame({"alpha":mixup_images_df["alpha_class_0"],"pred":mixup_images_df["predictions_0"]}).sort_values("pred")
    bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    alpha_confidence_df["bins"] = pd.cut(alpha_confidence_df["pred"],bins)
    new_df = alpha_confidence_df.groupby("bins").mean()
    mean_alphas = new_df["alpha"]

    axs[1].scatter(mixup_images_df["predictions_0"],mixup_images_df["alpha_class_0"], alpha = 0.01, label = "Mixup Data")
    axs[1].scatter(boundary_images_df["predictions_0"],boundary_images_df["alpha_class_0"], alpha = 0.1, label = "Boundary Points")

    for i in range(10):
        plt.axvline(i/10,linestyle="--",color = "black",alpha=0.1)

    middle_bins = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
    axs[1].plot(middle_bins,mean_alphas,marker = ".",label = "Mean Alpha",color = "black")
    axs[1].plot([0,1],[0,1],linestyle="--",color = "black",alpha=0.5)
    axs[1].set_ylabel(f"Alpha Value (Class: {positive_class})")
    axs[1].set_xlabel(f"Prediction Confidence (Class: {positive_class})")

    plt.legend(loc="lower right")

    axs[0].set_xlabel(f"Mean Predicted Probability (Positive Class: {positive_class})")
    axs[0].set_ylabel(f"Fraction of Positives (Positive Class: {positive_class})")
    axs[0].set_title("Calibration Curve")

    axs[0].plot(prob_pred_A,prob_demented_A,marker = ".", label = "Test Set A")
    axs[0].plot(prob_pred_B,prob_demented_B,marker = ".", label = "Test Set B")
    axs[0].plot(prob_pred_mix,prob_demented_mix,marker = ".", label = "Mixup")
    axs[0].plot([0, 1], [0, 1], linestyle='--',color="black", label = "Perfect Calibration")
    axs[0].legend()

    plt.show()

def muba_curves(mixup_errors,boundary_images_df, average = True, bins = 20, xlabel = "Proportion Class 0"):
    fig, axs = plt.subplots(1,2, figsize = (10,5))

    axs[0].hist(mixup_errors["alpha_class_0"],bins=[i/bins for i in range(bins+1)],)
    axs[0].set_ylabel("N")
    axs[0].set_xlabel(xlabel)
    axs[0].set_title("Error Distribution Curve")

    axs[1].hist(boundary_images_df["alpha_class_0"],bins=[i/bins for i in range(bins+1)],color="green")
    axs[1].set_ylabel("N")
    axs[1].set_xlabel(xlabel)
    axs[1].set_title("Boundary Distribution Curve")

    if average:
        axs[0].axvline(np.mean(mixup_errors["alpha_class_0"]), linestyle = "--",color="black", label = f"Mean: {round(np.mean(mixup_errors['alpha_class_0']),2)}")
        axs[0].legend()
        axs[1].axvline(np.mean(boundary_images_df["alpha_class_0"]), linestyle = "--",color="black", label = f"Mean: {round(np.mean(boundary_images_df['alpha_class_0']),2)}")
        axs[1].legend()

    plt.show()