#Date:2023-10-31
import numpy as np


def area_ave_rmse(data_real: np.array, data_predict: np.array, lat:np.array, lon:np.array) -> np.array:
    """_summary_

    Args:
        data_real (np.array): real data of shape (time,day,lev,lat,lon)
        data_predict (np.array): predict data of shape (time,day,lev,lat,lon)
        lat (np.array): lat of shape (lat,) degree

    Returns:
        np.array: rmse of shape (time,day,lev)
    """
    mask = np.array(np.isnan(data_real[0,0,0]))
    angle_radians = np.radians(lat)
    angle_radians = angle_radians[:, np.newaxis]
    angle_radians = np.tile(angle_radians, [1, len(lon)])
    angle_radians_mask = angle_radians
    angle_radians_mask[mask] = np.nan
    weights = np.cos(angle_radians_mask)/ np.nansum(np.cos(angle_radians_mask))
    
    data_diff = data_predict - data_real
    data_diff2 = data_diff**2
    data_diff2 = data_diff2*weights
    data_diff2 = data_diff2.reshape(data_diff.shape[0], data_diff.shape[1], data_diff.shape[2], data_diff.shape[3]*data_diff.shape[4])
    
    mse = np.nansum(data_diff2, axis=3)
    rmse = np.sqrt(mse)
    return rmse



def area_point_rmse(data_real: np.array, data_predict: np.array) -> np.array:
    """_summary_

    Args:
        data_real (np.array): real data of shape (time,day,lev,lat,lon)
        data_predict (np.array): predict data of shape (time,day,lev,lat,lon)

    Returns:
        np.array: rmse of shape (day,lev,lat,lon)
    """
    data_diff = data_predict - data_real
    data_diff2 = data_diff**2
    mse = np.nanmean(data_diff2,axis=0)
    rmse = np.sqrt(mse)
    return rmse

def area_point_r(data_real: np.array, data_predict: np.array) -> np.array:
    """_summary_

    Args:
        data_real (np.array): real data of shape (time,day,lev,lat,lon)
        data_predict (np.array): predict data of shape (time,day,lev,lat,lon)

    Returns:
        np.array: r of shape (day,lev,lat,lon)
    """
    r = np.zeros(shape=(data_real.shape[1], data_real.shape[2], data_real.shape[3], data_real.shape[4]))
    for i in range(data_real.shape[1]):
        print(f'lead {i} day')
        for j in range(data_real.shape[2]):
            for k in range(data_real.shape[3]):
                for l in range(data_real.shape[4]):
                    sample_x = data_real[:, i, j, k, l]
                    sample_y = data_predict[:, i, j, k, l]
                    r[i,j,k,l] = np.corrcoef(sample_x, sample_y)[0,1]

    return r

def area_ave_r(data_real: np.array, data_predict: np.array) -> np.array:
    """_summary_

    Args:
        data_real (np.array): real data of shape (time,day,lev,lat,lon)
        data_predict (np.array): predict data of shape (time,day,lev,lat,lon)

    Returns:
        np.array: r of shape (time, day, lev)
    """
    r = np.zeros(shape=(data_real.shape[0], data_real.shape[1], data_real.shape[2]))
    for i in range(data_real.shape[0]):
        print(f'lead {i} day')
        for j in range(data_real.shape[1]):
            for k in range(data_real.shape[2]):
                data_a = data_real[i, j, k, :, :]
                data_b = data_predict[i, j, k, :, :]
                valid_indices = ~np.isnan(data_a) & ~np.isnan(data_b)
                cleaned_data_a = data_a[valid_indices]
                cleaned_data_b = data_b[valid_indices]
                    
                r[i,j,k] = np.corrcoef(cleaned_data_a, cleaned_data_b)[0,1]

    return r

def area_ave_bias(data_real: np.array, data_predict: np.array, lat: np.array, lon: np.array) -> np.array:
    """_summary_

    Args:
        data_real (np.array): real data of shape (time,day,lev,lat,lon)
        data_predict (np.array): predict data of shape (time,day,lev,lat,lon)

    Returns:
        np.array: rmse of shape (time,day,lev)
    """
    mask = np.array(np.isnan(data_real[0,0,0]))
    angle_radians = np.radians(lat)
    angle_radians = angle_radians[:, np.newaxis]
    angle_radians = np.tile(angle_radians, [1, len(lon)])
    angle_radians_mask = angle_radians
    angle_radians_mask[mask] = np.nan
    weights = np.cos(angle_radians_mask)/ np.nansum(np.cos(angle_radians_mask))
    
    data_diff = data_predict - data_real
    data_diff = data_diff*weights
    
    data_diff = data_diff.reshape(data_diff.shape[0],data_diff.shape[1],data_diff.shape[2],data_diff.shape[3]*data_diff.shape[4])
    bias = np.nanmean(data_diff,axis=3)
    return bias

def area_point_bias(data_real: np.array, data_predict: np.array) -> np.array:
    """_summary_

    Args:
        data_real (np.array): real data of shape (time,day,lev,lat,lon)
        data_predict (np.array): predict data of shape (time,day,lev,lat,lon)

    Returns:
        np.array: rmse of shape (day,lev,lat,lon)
    """
    data_diff = data_predict - data_real
    bias = np.nanmean(data_diff,axis=0)
    return bias
