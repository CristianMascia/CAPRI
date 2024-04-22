import os
from pathlib import Path

import pandas as pd

quit()
srs = [1, 5, 10]
users = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 50]
loads = ["normal", "stress_cart", "stress_shop"]

for sr in srs:
    path_exps = "experiments_sr_" + str(sr)
    Path("../data_new/" + path_exps).mkdir(parents=True, exist_ok=True)
    for u in users:
        path_exps_u = path_exps + "/users_" + str(u)
        Path("../data_new/" + path_exps_u).mkdir(parents=True, exist_ok=True)
        for l in loads:
            path_exps_l = path_exps_u + "/" + l
            Path("../data_new/" + path_exps_l).mkdir(parents=True, exist_ok=True)
            for rep in range(1, 5):
                df = pd.read_csv(path_exps_l + "/esec_{}.csv_stats.csv".format(rep))

                index_home = df.index[df['Name'] == "/"].tolist()
                if len(index_home) == 1:
                    df.loc[index_home[0], 'Name'] = "home"
                elif len(index_home) > 1:
                    print("ERRORE")
                    quit()

                index_viewCart = df.index[(df['Name'] == "/cart") & (df['Type'] == "GET")]
                if len(index_viewCart) == 1:
                    df.loc[index_viewCart[0], 'Name'] = "viewCart"
                elif len(index_viewCart) > 1:
                    print("ERRORE")
                    quit()

                index_addToCart = df.index[(df['Name'] == "/cart") & (df['Type'] == "POST")]
                if len(index_addToCart) == 1:
                    df.loc[index_addToCart[0], 'Name'] = "addToCart"
                elif len(index_addToCart) > 1:
                    print("ERRORE")
                    quit()

                index_checkout = df.index[df['Name'] == "/cart/checkout"]
                if len(index_checkout) == 1:
                    df.loc[index_checkout[0], 'Name'] = "checkout"
                elif len(index_checkout) > 1:
                    print("ERRORE")
                    quit()

                index_cartEmpty = df.index[df['Name'] == "/cart/empty"]
                if len(index_cartEmpty) == 1:
                    df.loc[index_cartEmpty[0], 'Name'] = "cartEmpty"
                elif len(index_cartEmpty) > 1:
                    print("ERRORE")
                    quit()

                index_logout = df.index[df['Name'] == "/logout"]
                if len(index_logout) == 1:
                    df.loc[index_logout[0], 'Name'] = "logout"
                elif len(index_logout) > 1:
                    print("ERRORE")
                    quit()

                index_setCurrency = df.index[df['Name'] == "/setCurrency"]
                if len(index_setCurrency) == 1:
                    df.loc[index_setCurrency[0], 'Name'] = "setCurrency"
                elif len(index_setCurrency) > 1:
                    print("ERRORE")
                    quit()

                index_viewProduct = df.index[df['Name'].str.startswith('/product/')].tolist()
                product_rows = df.loc[index_viewProduct,]
                product_new_row = {
                    "Type": "GET",
                    "Name": "viewProduct",
                    "Request Count": product_rows['Request Count'].sum(),
                    "Failure Count": product_rows['Request Count'].sum(),
                    "Median Response Time": product_rows['Median Response Time'].median(),
                    "Average Response Time": product_rows["Average Response Time"].mean(),
                    "Min Response Time": product_rows["Min Response Time"].min(),
                    "Max Response Time": product_rows["Min Response Time"].max(),
                    "Average Content Size": product_rows["Average Content Size"].mean(),
                    "Requests/s": product_rows['Requests/s'].sum(),
                    "Failures/s": product_rows["Failures/s"].sum(),
                    "50%": product_rows["50%"].mean(),
                    "66%": product_rows["66%"].mean(),
                    "75%": product_rows["75%"].mean(),
                    "80%": product_rows["80%"].mean(),
                    "90%": product_rows["90%"].mean(),
                    "95%": product_rows["95%"].mean(),
                    "98%": product_rows["98%"].mean(),
                    "99%": product_rows["99%"].mean(),
                    "99.9%": product_rows["99.9%"].mean(),
                    "99.99%": product_rows["99.99%"].mean(),
                    "100%": product_rows["100%"].mean()

                }

                ndf = df.drop(index_viewProduct)
                ndf.loc[min(index_viewProduct)] = product_new_row
                ndf = ndf.sort_index(axis=0).reset_index(drop=True)

                ndf.to_csv(path_exps_l + "/esec_{}.csv_stats.csv".format(rep), index=False)
                # df = pd.read_csv(path_exps_l + "/esec_{}.csv_stats.csv".format(rep))
