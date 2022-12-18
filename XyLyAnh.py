from tkinter import *
from PIL import Image,ImageTk
from tkinter import filedialog
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from tkinter import messagebox
def ChonAnh():
    global fln
    fln = filedialog.askopenfilename(initialdir=os.getcwd(),title="Select Image File",filetypes=(("JPG file","*.jpg"),\
        ("PNGfile","*.png"),("All Files","*.*")))
    img = Image.open(fln)
    img = ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image = img
    lbl.place(x=583,y=254)
def LowpassIdeal():
    #Tạo Vùng Vẽ Furie
    fig = plt.figure(figsize=(16,9))
    (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)= fig.subplots(3, 3)
    #B1: Đọc ảnh , đưa ảnh vào ma trận điểm ảnh  và hiển thị ảnh có kích thước MxN
    image = cv2.imread(fln,0)
    image = cv2.resize(src=image, dsize=(100,100))
    f = np.asarray(image)
    M, N = np.shape(f) 
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Bước 1: Ảnh gốc f(x,y)')
    ax1.axis('off')
    #B2: Mở rộng ảnh có kick thước PxQ : P = 2 * M ; Q = 2 * N
    P, Q = 2*M , 2*N
    shape = np.shape(f)
    f_xy_p = np.zeros((P, Q))
    f_xy_p[:shape[0], :shape[1]] = f
    ax2.imshow(f_xy_p, cmap='gray')
    ax2.set_title('Bước 2: Ảnh mở rộng fp(x,y)')
    ax2.axis('off')
    #B3: Nhân fp(x,y) với (-1)^(x+y) để dời F0 vào tâm ảnh
    F_xy_p = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            F_xy_p[x, y] = f_xy_p[x, y] * np.power(-1, x + y)
    ax3.imshow(F_xy_p, cmap='gray')
    ax3.set_title('Bước 3: Nhân fp(x,y) với (-1)^(x+y) để dời F0 vào tâm ảnh')
    ax3.axis('off')
    #B4: Biến đổi Fourier
    def DFT1D(img):
        U = len(img)
        outarry = np.zeros(U, dtype=complex)
        for m in range(U):
            sum = 0.0
            for n in range(U):
                e = np.exp(-1j * 2 * np.pi * m * n / U)
                sum += img[n] * e
            outarry[m] = sum
        return outarry

    def IDFT1D(img):
        U = len(img)
        outarry = np.zeros(U,dtype=complex)
        for n in range(U):
            sum = 0.0
            for m in range(U):
                e = np.exp(1j * 2 * np.pi * m * n / U)
                sum += img[m]*e
            pixel = sum/U
            outarry[n]=pixel
        return outarry
    dft_cot = dft_hang = np.zeros((P, Q))
    for i in range(P):
        dft_cot[i] = DFT1D(F_xy_p[i])
    for j in range(Q):
        dft_hang[:, j] = DFT1D(dft_cot[:, j])
    ax4.imshow(dft_hang, cmap='gray')
    ax4.set_title('Bước 4: Phổ tần số ảnh sau khi DFT')
    ax4.axis('off')
    #B5.1: Cho hàm lọc có giá trị thực H(u,v) đối xứng qua tâm
    def lowPass_Ideals(D0,U,V):
        # H is our filter
        H = np.zeros((U, V))
        D = np.zeros((U, V))
        U0 = int(U / 2)
        V0 = int(V / 2)
    # Tính khoảng cách
        for u in range(U):
            for v in range(V):
                u2 = np.power(u, 2)
                v2 = np.power(v, 2)
                D[u, v] = np.sqrt( u2 +v2)
    # Tính bộ lọc
        for u in range(U):
            for v in range(V):
                if D[np.abs(u - U0), np.abs(v - V0)] <= D0:
                    H[u, v] = 1
                else:
                    H[u, v] = 0
        return H
    H_uv = lowPass_Ideals(60,P,Q)
    #B5.2: Nhân ảnh sau khi DFT với ảnh sau khi lọc
    ax5.imshow(H_uv, cmap='gray')
    ax5.set_title('Bước 5.1: H(u,v) đối xứng qua tâm bộ lọc ')
    ax5.axis('off')
    G_uv = np.multiply(dft_hang, H_uv)
    ax6.imshow(G_uv, cmap='gray')
    ax6.set_title('Bước 5.2: Nhân F(u,v) * H(u,v)')
    ax6.axis('off')
    #B6.1: Biến đổi Fourier ngược
    idft_cot = idft_hang = np.zeros((P, Q))
    for i in range(P):
        idft_cot[i] = IDFT1D(G_uv[i])
    for j in range(Q):
        idft_hang[:, j] = IDFT1D(idft_cot[:, j])
    ax7.imshow(idft_hang, cmap='gray')
    ax7.set_title('Bước 6.1: Biến đổi Fourier ngược')
    ax7.axis('off')
    #B6.2: Lấy phần thực và dời về gốc tọa độ
    g_array = np.asarray(idft_hang.real)
    P, Q = np.shape(g_array)
    g_xy_p = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            g_xy_p[x, y] = g_array[x, y] * np.power(-1, x + y)
    ax8.imshow(g_xy_p, cmap='gray')
    ax8.set_title('Bước 6.2: Lấy phần thực và dời về gốc tọa độ')
    ax8.axis('off')
    #B7: Ảnh cuối cùng sau xử lý
    g_xy = g_xy_p[:shape[0], :shape[1]]
    ax9.imshow(g_xy, cmap='gray')
    ax9.set_title('Bước 7: Lấy phần thực và dời về gốc tọa độ')
    ax9.axis('off')
    plt.show()

root = Tk()
root.title("Xử Lý Ảnh Trên Miền Tần Số")
root.geometry("1366x768")

label = Label(root,text="Nhóm 15 - 70DCTT21")
label.pack(side=TOP,pady=25)

frm = Frame(root)
frm.pack(side=TOP , padx= 15 , pady= 15)

lbl = Label(root)
lbl.pack()

btn1 = Button(frm,text="Chọn Ảnh",command=ChonAnh)
btn1.pack(side=LEFT,padx=15)

btn2 = Button(frm,text="Lọc Thông Thấp Ideal",command=LowpassIdeal)
btn2.pack(side=LEFT,padx=15) 

btn3 = Button(frm,text="Lọc Thông Thấp GauSsian")
btn3.pack(side=LEFT,padx=15) 

btn4 = Button(frm,text="Lọc Thông Thấp ButterWorth")
btn4.pack(side=LEFT,padx=15) 

btn5 = Button(frm,text="Lọc Thông Cao Ideal")
btn5.pack(side=LEFT,padx=15) 

btn5 = Button(frm,text="Lọc Thông Cao GauSian")
btn5.pack(side=LEFT,padx=15) 

btn5 = Button(frm,text="Lọc Thông Cao ButterWorth")
btn5.pack(side=LEFT,padx=15) 

root.mainloop()
