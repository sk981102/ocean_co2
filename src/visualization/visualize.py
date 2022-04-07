import imageio
import matplotlib.colors as mcolors



def create_gifs(target_file_name, y_true, y_pred, X, conv_lstm = False):
    filenames = []
    
    y_true = np.flipud(y_true)
    y_pred = np.flipud(y_pred)
    
    if conv_lstm:
        for i in range(y_pred.shape[0]):
            figure, axis = plt.subplots(2, 2,figsize=(12, 6))

            img=axis[0][0].imshow(y_pred[i][1],cmap="coolwarm", interpolation="nearest")
            axis[0][0].set_title("prediction")
            plt.colorbar(img,ax=axis)

            img1=axis[0][1].imshow(y_true[i][1],cmap="coolwarm", interpolation="nearest")
            axis[0][1].set_title("true")

            diff=y_pred[i][1]-y_true[i][1]
            img2=axis[1][0].imshow(diff,cmap="RdBu", interpolation="nearest",norm=norm)
            axis[1][0].set_title("residual")
            plt.colorbar(img2,ax=axis)

            img2=axis[1][1].imshow(np.flipud(X[i][1][:,:,5]),cmap="coolwarm", interpolation="nearest")
            axis[1][1].set_title("input: previous pco2")
            # create file name and append it to a list
            filename = f'{i}.png'
            filenames.append(filename)

            # save frame
            plt.savefig(filename)
            plt.close()
        
    else:
        for i in range(y_pred.shape[0]):
            # plot the line chart
            figure, axis = plt.subplots(2, 2,figsize=(12, 6))

            img=axis[0][0].imshow(y_pred[i],cmap="coolwarm", interpolation="nearest")
            axis[0][0].set_title("prediction")
            plt.colorbar(img,ax=axis)

            img1=axis[0][1].imshow(y_true[i],cmap="coolwarm", interpolation="nearest")
            axis[0][1].set_title("true")

            diff=np.squeeze(y_pred[i]-y_true[i]))
            img2=axis[1][0].imshow(diff,cmap="RdBu", interpolation="nearest",norm=norm)
            axis[1][0].set_title("residual")
            plt.colorbar(img2,ax=axis)

            # create file name and append it to a list
            filename = f'{i}.png'
            filenames.append(filename)

            # save frame
            plt.savefig(filename)
            plt.close()
        
    # build gif
    with imageio.get_writer(target_file_name, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)
    
    pass


