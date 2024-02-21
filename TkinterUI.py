import tkinter
import customtkinter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from DEDlib import *
import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)

matplotlib.rc('axes',edgecolor='white')

def counter():#N,Nmax
    global paused,stopped,started,DEDargs,pbar,Nfin
    #print(count_var.get())
    if not paused and pbar.n!=DEDargs[0]:
        #N+=1
        #pbar.n=N
        #pbar.refresh()
        iterationDED()
        count_var.set(f"{pbar.n}/{DEDargs[0]}")
        progressbar.set(pbar.n/DEDargs[0])
    if not stopped and pbar.n!=DEDargs[0]:
        #pbar.refresh()
        app.after(1,counter)
    elif pbar.n==DEDargs[0]:
        pbar.close()

    else:
        stopped=False
        started=False
        count_var.set(f"{0}/{DEDargs[0]}")
        progressbar.set(0)
        pbar.n=0
        #Nfin=np.zeros(len(DEDargs[11]),dtype='float')
        #pbar.refresh()
        pbar.reset()

def startDED():
    global started,paused,number,istar
    istar+=1
    if istar==1:
        if not started: 
            started=True
            #print("start")
        counter()


def pauseDED():
    global paused,started
    if started:
        paused = not paused 

def stopDED():
    global stopped,started,paused,number,istar
    if not stopped and started: 
        stopped=True
        paused=False
        #print("stop")
        number=0
        istar=0

def DOSplottest(fDOS,Lor,omega,name,labels,log=False,ymax=1.2,save=True):
    """DOSplot(fDOS,Lor,omega,name,labels). 
A plot function to present results from the AIM moddeling for a single results with a comparison to the non-interacting DOS."""
    fig=plt.figure(figsize=(10,8))
    plt.rc('legend',fontsize=17)
    plt.rc('font',size=25)
    plt.rc('xtick',labelsize=25)
    plt.rc('ytick',labelsize=25)
    axis_font={'fontname':'Calibri','size':'25'}
    plt.xlim(min(omega),max(omega))
    if not log:
        plt.gca().set_ylim(bottom=0,top=ymax)
        plt.gca().set_xticks(np.linspace(min(omega),max(omega),2*int(max(omega))+1),minor=False)
    else: 
        plt.yscale('log')
        plt.gca().set_ylim(bottom=0.0001,top=10)
        plt.gca().set_xticks([-8,-6,-4,-2,0,2,4,6,8],minor=False)
    plt.xlabel("$\\omega$ [-]",**axis_font)
    plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom",rotation=0,labelpad=30,**axis_font)
    plt.plot(omega,Lor,'--r',linewidth=4,label='$\\rho_0$')
    plt.plot(omega,fDOS,'-b',label=labels)
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(name+'.png',format='png')
        plt.savefig(name+'.svg',format='svg',dpi=3600)
    plt.pause(5)
    plt.close()
    return fig

def showgraph():
    global omega,Lor,AvgSigmadat,Nfin,DEDargs,app
    fDOS=(-np.imag(np.nan_to_num(1/(omega-AvgSigmadat/Nfin[:,None]-DEDargs[4]+1j*DEDargs[5])))/np.pi).squeeze()
    #fig=DOSplottest(fDOS,Lor,omega,'constraintN5p','$\\rho,$n='+str(DEDargs[1]),save=False)
    fig=plt.figure(figsize=(7,5.6))#,facecolor='#242424')

    plt.rc('legend',fontsize=12)
    plt.rc('font',size=17)
    plt.rc('xtick',labelsize=17,color='white')
    plt.rc('ytick',labelsize=17,color='white')
    axis_font={'fontname':'Calibri','size':'17'}
    plt.xlim(min(omega),max(omega))
    plt.gca().set_ylim(bottom=0,top=1.2)
    plt.gca().set_xticks(np.linspace(min(omega),max(omega),2*int(max(omega))+1),minor=False)
    plt.xlabel("$\\omega$ [-]",**axis_font,color='white')
    plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom",rotation=0,labelpad=30,**axis_font,color='white')
    plt.plot(omega,Lor,'--r',linewidth=4,label='$\\rho_0$')
    plt.plot(omega,fDOS,'-b',label='$\\rho,$n='+str(DEDargs[1]))
    plt.legend(fancybox=False).get_frame().set_edgecolor('black')
    plt.grid()
    plt.tight_layout()
    fig.set_facecolor("none")

    #ax = plt.axes()

    plt.gca().set_facecolor("#242424")

    canvas = FigureCanvasTkAgg(fig,master=app)

    s = tkinter.ttk.Style()
    bg = s.lookup("TFrame", "background")
    bg_16bit =app.winfo_rgb('DarkGray')#bg
    bg_string="#242424"

    #bg_string = "#" + "".join([hex(bg_color >> 8)[2:] for bg_color in bg_16bit])
    canvas.get_tk_widget().config(bg=bg_string)

    canvas.draw()
    canvas.get_tk_widget().grid(row=6,columnspan=3)#.get_tk_widget()
    app.update()

#global paused,started,stopped
#paused=False
#started=False
#stopped=False
#istar=0
#number=0

# System Settings
#customtkinter.set_appearance_mode("dark")#"System"
#customtkinter.set_default_color_theme("blue")

# Our app frame
#app=customtkinter.CTk()
#app.geometry("500x300")#("720x480")
#app.minsize(300, 200)
#app.title("Distributional Exact Diagonalization AIM simulator")

# Adduing UI Elements
#title=customtkinter.CTkLabel(app,text="Choose DED data file directory")
#title.pack(padx=10,pady=10)
#title.grid(row=0, column=0,columnspan=3)

# Directory input
#dir_var=tkinter.StringVar()
#dir=customtkinter.CTkEntry(app, width=350,height=40,textvariable=dir_var)
#dir.pack()
#dir.grid(row=1, column=0,columnspan=3)


#progressbar=customtkinter.CTkProgressBar(app)
#progressbar.pack(padx=10,pady=10,side = "left")
#progressbar.grid(row=2, column=0,padx=10,pady=10, columnspan=2, sticky="nsew")
#progressbar.set(0)

# Count display
#count_var=tkinter.StringVar()
#count_var.set(f"{number}/{Nmax}")#
#count_text=customtkinter.CTkLabel(app,textvariable=count_var)
#count_text.pack(side = "left")
#count_text.grid(row=2, column=2,padx=10,pady=10, sticky="nsew")

# Start and pause button
#startb=customtkinter.CTkButton(app,text="Start",command=startDED)#command=lambda: startDED(started))
#startb.pack(padx=10,pady=10,side = "left")
#startb.grid(row=3, column=0,padx=10,pady=10, sticky="ew")

#pauseb=customtkinter.CTkButton(app,text="Pause",command=pauseDED)#command=lambda: pauseDED(paused,started))
#pauseb.pack(padx=10,pady=10,side="left")
#pauseb.grid(row=3, column=1,padx=10,pady=10, sticky="ew")

#stopb=customtkinter.CTkButton(app,text="Stop",command=stopDED)#command=lambda: stopDED(started))
#stopb.pack(padx=10,pady=10,side = "left")
#stopb.grid(row=3, column=2,padx=10,pady=10, sticky="ew")

#print(startb["state"])


# Run app

def DEDUI():
    global paused,started,stopped,istar,number,count_var,progressbar,app,DEDargs
    paused=False
    started=False
    stopped=False
    istar=0
    number=0
    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("blue")
    app=customtkinter.CTk()
    app.geometry("500x600")
    app.minsize(300, 200)

    #app.resizable(0, 0)

    app.columnconfigure(0, weight=1)
    #app.rowconfigure(0, weight=1)

    app.after(201, lambda :app.iconbitmap('DEDicon.ico'))

    app.title("Distributional Exact Diagonalization AIM simulator")
    title=customtkinter.CTkLabel(app,text="Choose DED data file directory")
    title.grid(row=0, column=0,columnspan=3)
    dir_var=tkinter.StringVar()
    dir=customtkinter.CTkEntry(app, width=350,height=40,textvariable=dir_var)
    dir.grid(row=1, column=0,columnspan=3)
    progressbar=customtkinter.CTkProgressBar(app)
    progressbar.grid(row=2, column=0,padx=10,pady=10, columnspan=2, sticky="nsew")
    progressbar.set(0)
    count_var=tkinter.StringVar()
    count_var.set(f"{number}/{DEDargs[0]}")
    count_text=customtkinter.CTkLabel(app,textvariable=count_var)
    count_text.grid(row=2, column=2,padx=10,pady=10, sticky="nsew")
    startb=customtkinter.CTkButton(app,text="Start",command=startDED)
    startb.grid(row=3, column=0,padx=10,pady=10, sticky="ew")
    pauseb=customtkinter.CTkButton(app,text="Pause",command=pauseDED)
    pauseb.grid(row=3, column=1,padx=10,pady=10, sticky="ew")
    stopb=customtkinter.CTkButton(app,text="Stop",command=stopDED)
    stopb.grid(row=3, column=2,padx=10,pady=10, sticky="ew")
    stopb=customtkinter.CTkButton(app,text="Show Results",command=showgraph)
    stopb.grid(row=5, column=1,padx=10,pady=10, sticky="ew")
    #fig=plt.figure(figsize=(7,5.6))
    #canvas = FigureCanvasTkAgg(fig,master=app)
    #canvas.draw()
    #canvas.get_tk_widget().grid(row=6,columnspan=3)
    frame=customtkinter.CTkFrame(app,width=2.2*app.winfo_width(),height=1.8*app.winfo_height())
    frame.grid(row=6,columnspan=3)
    return app


def main(N=200000,poles=2,U=3,Sigma=3/2,Ed=-3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39],ctype='n',Edcalc='',bound=3,Tk=[0],Nimpurities=1,U2=0,J=0,posb=1,log=False,base=1.5):
    global omega,selectpcT,selectpT,Npoles,c,pbar,eta,Hn,n,AvgSigmadat,Nfin,nd,DEDargs,Lor
    DEDargs=[N,poles,U,Sigma,Ed,Gamma,ctype,Edcalc,Nimpurities,U2,J,Tk]
    if log: omega,selectpcT,selectpT,Npoles=np.concatenate((-np.logspace(np.log(bound)/np.log(base),np.log(1e-5)/np.log(base),int(np.round(SizeO/2)),base=base),np.logspace(np.log(1e-5)/np.log(base),np.log(bound)/np.log(base),int(np.round(SizeO/2)),base=base))),[],[],int(poles/Nimpurities)
    else: omega,selectpcT,selectpT,Npoles=np.linspace(-bound,bound,SizeO),[],[],int(poles/Nimpurities)
    c,pbar,eta=[Jordan_wigner_transform(i,2*poles) for i in range(2*poles)],trange(N,position=posb,leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),etaco[0]*abs(omega)+etaco[1]
    (Hn,n),AvgSigmadat,Nfin,nd=Operators(c,Nimpurities,poles),np.zeros((len(Tk),SizeO),dtype='complex_'),np.zeros(len(Tk),dtype='float'),np.zeros(len(Tk),dtype='complex_')
    Lor=Lorentzian(omega,DEDargs[5],DEDargs[1],DEDargs[4],DEDargs[3])[0]
    DEDUI().mainloop()


def iterationDED(reset=False):
    global omega,selectpcT,selectpT,Npoles,c,pbar,eta,Hn,n,AvgSigmadat,Nfin,nd
    while not reset:
        NewM,nonG,select=Startrans(Npoles,np.sort(Lorentzian(omega,DEDargs[5],Npoles,DEDargs[4],DEDargs[3])[1]),omega,eta)
        H0,H=HamiltonianAIM(np.repeat(NewM[0][0],DEDargs[8]),np.tile([NewM[k+1][k+1] for k in range(len(NewM)-1)],(DEDargs[8],1)),np.tile(NewM[0,1:],(DEDargs[8],1)),DEDargs[2],DEDargs[3],DEDargs[9],DEDargs[10],Hn)
        try: (MBGdat,Boltzmann,Ev0),reset=Constraint(DEDargs[6],H0,H,omega,eta,c,n,DEDargs[11],np.array([ar<DEDargs[0] for ar in Nfin]))
        except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence): (MBGdat,Boltzmann,Ev0),reset=(np.zeros(len(omega),dtype='complex_'),np.zeros(len(DEDargs[11])),np.array([])),False
        if np.isnan(1/nonG-1/MBGdat+DEDargs[3]).any() or np.array([i>=1000 for i in np.real(1/nonG-1/MBGdat+DEDargs[3])]).any(): reset=False
        selectpT.append(select)
    Nfin,AvgSigmadat,nd=Nfin+Boltzmann,AvgSigmadat+(1/nonG-1/MBGdat+DEDargs[3])*Boltzmann[:,None],nd+np.conj(Ev0).T@sum(Hn[0]).data.tocoo()@Ev0*Boltzmann
    selectpcT.append(select)
    if DEDargs[6]=='sn': pbar.n+=1
    else: pbar.n=int(min(Nfin))
    pbar.refresh()
    #print(pbar.n,Boltzmann,Boltzmann+Nfin)
main(N=10000)