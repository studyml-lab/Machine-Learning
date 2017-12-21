from sklearn import linear_model

house_train = pd.read_csv('e:/kaggle/homeprices/train.csv')
house_train.shape
house_train.info()

house_test = pd.read_csv('e:/kaggle/homeprices/test.csv')
house_test.shape
house_test.info()

dims= ['LotArea']
X_train = house_train[dims]
y_train = house_train['SalePrice']

lr1 = linear_model.LinearRegression()
lr2 = linear_model.Ridge
lr2 = linear_model.

# fit call actual learning prrocess
lr1.fit(X_train,y_train)
lr1.coef_
lr1.intercept_
lr1.fit_intercept

X_test = house_test[dims]
house_test['SalePrice'] = lr1.predict(X_test)

house_test.to_csv('house1.csv', columns=['Id','SalePrice'],index=False)




Question 1 : 
Question 2 : cat -> continuous conversion


