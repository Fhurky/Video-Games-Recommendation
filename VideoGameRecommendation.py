import numpy as np

veri = np.array([[1,2,3],[4,5,6],[3,2,1]])

# Verinin kaç boyutlu olduğunu öğrenmek için
#print(veri.ndim)

# Verinin boyutlarını öğrenmek için
#print(veri.shape)

# Verinin içerdiği elementlerinin tipini öğrenmek için
#print(veri.dtype)

# Verinin tipini öğrenmek için
#print(type(veri))

# Veri Tipleri ----------------------------------------------------------------------------------------------------------------

tamsayi = np.array([1,2,3], dtype=np.int_) # tamsayi = np.array([1,2,3], dtype=int)
#print(tamsayi.dtype)

mantiksal = np.array([-1,0,1,2], dtype=np.bool_) # mantiksal = np.array([-1,0,1,2], dtype=bool)
#print(mantiksal.dtype)

ondalik = np.array([100.1,200.1,300.1], dtype=np.float_) # ondalik = np.array([100.1,200.1,300.1], dtype=float)
#print(ondalik.dtype)

komplex = np.array([-1,0,1], dtype=np.complex_) # komplex = np.array([-1,0,1], dtype=complex) // komplex.real // komplex.imag
#print(komplex.dtype)

# Dizi Oluşturma ---------------------------------------------------------------------------------------------------------------

array = np.zeros((2)) # 1 boyutlu sifirlardan olusan bir sabit
array = np.ones((2)) # 1 boyutlu birlerden olusan bir sabit
array = np.zeros((1,2,3)) # 3 boyutlu sifirlardan olusan bir dizi

array = np.full((2,2), 7) # 7 lerden olusan 2x2 lik bir matrix
array = np.diag((4,4), 1) # Diagonali 1 lerden ve geri kalanı 0 lardan olusan bir matrix

x = np.arange(9).reshape((3,3))
x = np.diag(np.diag(x, 0)) # Köşegen haricindekileri sifirlama islemi (0 degeri kosegen konumudur)
#print(x)

array = np.eye(3,3) # 3x3 ve diagonali 1 olan ve geri kalani 0 olan matrix
#print(array)

array = np.arange(start=1,stop=10,step=3) # 1 den baslayarak 3er artacak sekilde 10a kadar(10 dahil degil) dizi olustur
#print(array)

array = np.linspace(start=1, stop=10, num=8) # 1 den baslayarak 10a kadar(10 dahil) 8 adet esit aralikli sayi dizisi olustur
#print(array)

array = np.empty([2,2]) # rastgele degerli 2x2 lik bir matrix olustur
#print(array)

array = np.random.seed(0) # Seed sabitle
array = np.random.rand(2,3) # 2x3 lük rastgele dizi (0-1 arasında)
#print(array)

array = np.arange(start=0, stop=101, step=10)
#print(array[0:10:1]) 

array = np.array([[1,2,3],[4,5,6],[7,8,9]])
#print(type(array[2,-1]))

array = np.array((1,2,3)) # Diziye eleman ekleme
array = np.append(array, (4,5,6))
#print(array)

array = np.delete(array, (1,2)) # 1 nolu indisteki elemani siler
#print(array)

array = np.ndarray.reshape(array, (2,2)) # 2x2 matrix haline getirmek
#print(array)