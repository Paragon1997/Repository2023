import tkinter as tk
import customtkinter as ctk
from CTkMessagebox import CTkMessagebox
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from DEDlib import *
import warnings
import json
from json import JSONEncoder
from ast import literal_eval
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
warnings.filterwarnings("ignore",category=RuntimeWarning)

#prevent cross sim loading
#add ask window main root window
#other main functions in seperate windows (graphene, S etc.)
#make exe with pyinstaller

class NumpyArrayEncoder(JSONEncoder):
    def default(self,obj):
        if isinstance(obj,np.ndarray): return obj.tolist()
        return JSONEncoder.default(self,obj)
    
def AvgSigmajsonfileR(name):
    data=json.load(open(name))
    data["AvgSigmadat"],data["nd"]=np.array(data["AvgSigmadat"],dtype=object).astype(np.complex128),np.array(data["nd"],dtype=object).astype(np.complex128)
    return data

def AvgSigmajsonfileW(root,name):
    if root.DEDargs[7]=='AS':root.fDOS=(-np.imag(np.nan_to_num(1/(root.omega-root.AvgSigmadat/root.Nfin[:,None]+(root.AvgSigmadat[:,int(np.round(root.DEDargs[13]/2))]/root.Nfin)[:,None]+1j*root.DEDargs[5])))/np.pi).squeeze()
    else:root.fDOS=(-np.imag(np.nan_to_num(1/(root.omega-root.AvgSigmadat/root.Nfin[:,None]-root.DEDargs[4]+1j*root.DEDargs[5])))/np.pi).squeeze()
    data={"Ntot":root.pbar.total,"Nit":root.pbar.n,"telapsed":root.pbar.format_dict["elapsed"],"poles":root.DEDargs[1],"U":root.DEDargs[2],"Sigma":root.DEDargs[3],"Ed":root.DEDargs[4],"Gamma":root.DEDargs[5],"ctype":root.DEDargs[6],"Edcalc":root.DEDargs[7],"Nimpurities":root.DEDargs[8],"U2":root.DEDargs[9],"J":root.DEDargs[10],"Tk":root.DEDargs[11],"etaco":root.DEDargs[12],"SizeO":root.DEDargs[13],"bound":root.DEDargs[14],"posb":root.DEDargs[15],"log":root.DEDargs[16],"base":root.DEDargs[17],
    "Nfin":root.Nfin,"omega":root.omega,"fDOS":root.fDOS,"AvgSigmadat":[str(i) for i in (root.AvgSigmadat/root.Nfin[:,None]).squeeze()],"nd":[str(i) for i in (root.nd/root.Nfin)]}
    jsonObj=json.dumps(data,cls=NumpyArrayEncoder)
    with open(name,"w") as outfile: outfile.write(jsonObj)

def savfilename(entry):
    if not entry.get().endswith(".json"):
        entry.delete(0,last_index=tk.END)
        entry.insert(0,'Try again')

def savedata(root,entry):
    if root.started or root.stopped:
        try:
            if not entry.get().endswith(".json"):
                entry.delete(0,last_index=tk.END)
                entry.insert(0,'Try again')
                return False
            else:   
                AvgSigmajsonfileW(root,entry.get())
                return True
        except IOError:
            entry.delete(0,last_index=tk.END)
            entry.insert(0,'Try again')
            return False

def savegraph(root,entry):
    mpl.rc('axes',edgecolor='black')
    _=savedata(root,entry)
    if root.DEDargs[7]=='AS':root.fDOS=(-np.imag(np.nan_to_num(1/(root.omega-root.AvgSigmadat/root.Nfin[:,None]+(root.AvgSigmadat[:,int(np.round(root.DEDargs[13]/2))]/root.Nfin)[:,None]+1j*root.DEDargs[5])))/np.pi).squeeze()
    else:root.fDOS=(-np.imag(np.nan_to_num(1/(root.omega-root.AvgSigmadat/root.Nfin[:,None]-root.DEDargs[4]+1j*root.DEDargs[5])))/np.pi).squeeze()
    if entry.get().endswith(".json"):
        DOSplot(root.fDOS,root.Lor,root.omega,entry.get().replace(".json",""),'$\\rho_{constr.},N,$n='+str(root.DEDargs[1]))
    else:
        entry.delete(0,last_index=tk.END)
        entry.insert(0,'Try again') 

class ProgressBar(ctk.CTkProgressBar):
    def __init__(self,itnum,Total,*args,**kwargs):
        self.itnum,self.Total=itnum,Total
        super().__init__(*args,**kwargs)
        self._canvas.create_text(0,0,text=f"{itnum}/{Total}",fill="white",font=14,anchor="c",tags="progress_text")

    def _update_dimensions_event(self,event):
        super()._update_dimensions_event(event)
        self._canvas.coords("progress_text",event.width/2,event.height/2)

    def set(self,val,**kwargs):
        super().set(val,**kwargs)
        self._canvas.itemconfigure("progress_text",text=f"{self.itnum}/{self.Total}")

def CenterWindowToDisplay(Screen:ctk.CTk,width:int,height:int,scale_factor:float=1.0): return f"{width}x{height}+{int((0.5*(Screen.winfo_screenwidth()-width)-7)*scale_factor)}+{int((0.5*(Screen.winfo_screenheight()-height)-31)*scale_factor)}"

class mainApp(ctk.CTk):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.app_width,self.app_height=300,200
        self.geometry(CenterWindowToDisplay(self,self.app_width,self.app_height,self._get_window_scaling()))
        self.iconbitmap('DEDicon.ico')
        self.resizable(width=False,height=False)
        self.title("DED simulator menu")
        self.focus()
        self.grid_rowconfigure((0,1,2),weight=0)
        self.grid_columnconfigure(0,weight=1)
        self.menu_label=ctk.CTkLabel(self,text="Choose simulation type",font=ctk.CTkFont(size=20,weight="bold"))
        self.menu_label.grid(row=0,column=0,padx=20,pady=(20,10))
        self.scaling_optionemenu=ctk.CTkOptionMenu(self,width=200,values=["SAIM single sim","Sampled poles Distr.","ASAIM single sim","GNR SAIM single sim","Impurity Entropy","Stdev calculator"])
        self.scaling_optionemenu.grid(row=1,column=0,padx=20,pady=(5,0))
        self.button_open=ctk.CTkButton(self,text="Open",command=lambda:self.open_toplevel(simsel=self.scaling_optionemenu.get()))
        self.button_open.grid(row=2,column=0,padx=20,pady=(20,20))
        self.top_level_windows={"SAIM single sim":SAIMWINDOW,"Sampled poles Distr.":polesWINDOW,"ASAIM single sim":ASAIMWINDOW,"GNR SAIM single sim":GNRWINDOW,"Impurity Entropy":EntropyWINDOW,"Stdev calculator":StdevWINDOW}
        self.protocol('WM_DELETE_WINDOW',self.quitApp)

    def open_toplevel(self,simsel):
        self.toplevel_window=self.top_level_windows[simsel](selfroot=self)
        self.after(100,self.lower)
        self.scaling_optionemenu.configure(state="disabled")
        self.button_open.configure(state="disabled")

    def quitApp(self):
        self.msg=CTkMessagebox(master=self,title="Exit?",message="Are you sure you want to quit the entire application?",icon="question",option_1="Cancel",option_2="No",option_3="Yes")
        if self.msg.get()=="Yes": self.destroy()

class SAIMWINDOW(ctk.CTkToplevel):
    def __init__(self,selfroot,N=200000,poles=4,U=3,Sigma=3/2,Ed=-3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39],ctype='n',Edcalc='',bound=3,Tk=[0],Nimpurities=1,U2=0,J=0,posb=1,log=False,base=1.5,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.root,self.paused,self.started,self.stopped,self.loaded,self.telapsed,self.DEDargs=selfroot,False,False,False,False,0,[N,poles,U,Sigma,Ed,Gamma,ctype,Edcalc,Nimpurities,U2,J,Tk,etaco,SizeO,bound,posb,log,base]
        if self.DEDargs[16]:self.omega,self.Npoles=np.concatenate((-np.logspace(np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),np.log(1e-5)/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]),np.logspace(np.log(1e-5)/np.log(self.DEDargs[17]),np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]))),int(self.DEDargs[1]/self.DEDargs[8])
        else:self.omega,self.Npoles=np.linspace(-self.DEDargs[14],self.DEDargs[14],self.DEDargs[13]),int(self.DEDargs[1]/self.DEDargs[8])
        self.c,self.pbar,self.eta=[Jordan_wigner_transform(i,2*self.DEDargs[1]) for i in range(2*self.DEDargs[1])],trange(self.DEDargs[0],position=self.DEDargs[15],leave=False,desc='Iterations',bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'),self.DEDargs[12][0]*abs(self.omega)+self.DEDargs[12][1]
        (self.Hn,self.n),self.AvgSigmadat,self.Nfin,self.nd,self.Lor=Operators(self.c,self.DEDargs[8],self.DEDargs[1]),np.zeros((len(self.DEDargs[11]),self.DEDargs[13]),dtype='complex_'),np.zeros(len(self.DEDargs[11]),dtype='float'),np.zeros(len(self.DEDargs[11]),dtype='complex_'),Lorentzian(self.omega,self.DEDargs[5],self.DEDargs[1],self.DEDargs[4],self.DEDargs[3])[0]
        self.title("Distributional Exact Diagonalization AIM simulator")
        self.app_width,self.app_height=1100,580
        self.geometry(CenterWindowToDisplay(self,self.app_width,self.app_height,self._get_window_scaling()))
        self.after(200,lambda:self.iconbitmap('DEDicon.ico'))
        self.resizable(width=False,height=False)
        self.after(200,self.focus)
        self.grid_columnconfigure(1,weight=1)
        self.grid_columnconfigure((2,3),weight=0)
        self.grid_rowconfigure((0,1,2),weight=1)
        self.sidebar_frame=ctk.CTkFrame(self,width=140,corner_radius=0)
        self.sidebar_frame.grid(row=0,column=0,rowspan=5,sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(16,weight=1)
        self.logo_label=ctk.CTkLabel(self.sidebar_frame,text="AIM Parameters",font=ctk.CTkFont(size=20,weight="bold"))
        self.logo_label.grid(row=0,column=0,padx=20,pady=(20,10))
        self.N_label=ctk.CTkLabel(self.sidebar_frame,text="No. of iterations DED:",anchor="w")
        self.N_label.grid(row=1,column=0,padx=20,pady=(5,0))
        self.N_Entry=ctk.CTkEntry(self.sidebar_frame,placeholder_text="200000")
        self.N_Entry.grid(row=2,column=0,padx=20,pady=(0,0))
        self.U_label=ctk.CTkLabel(self.sidebar_frame,text="Coulomb repulsion U:",anchor="w")
        self.U_label.grid(row=3,column=0,padx=20,pady=(5,0))
        self.U_Entry=ctk.CTkEntry(self.sidebar_frame,placeholder_text="3")
        self.U_Entry.grid(row=4,column=0,padx=20,pady=(0,0))
        self.Sigma_label=ctk.CTkLabel(self.sidebar_frame,text="Effective one-body potential \u03A3\N{SUBSCRIPT ZERO}:",anchor="w")
        self.Sigma_label.grid(row=5,column=0,padx=20,pady=(5,0))
        self.Sigma_Entry=ctk.CTkEntry(self.sidebar_frame,placeholder_text="1.5")
        self.Sigma_Entry.grid(row=6,column=0,padx=20,pady=(0,0))
        self.Ed_label=ctk.CTkLabel(self.sidebar_frame,text="Electron impurity energy \u03B5d:",anchor="w")
        self.Ed_label.grid(row=7,column=0,padx=20,pady=(5,0))
        self.Ed_Entry=ctk.CTkEntry(self.sidebar_frame,placeholder_text="-1.5")
        self.Ed_Entry.grid(row=8,column=0,padx=20,pady=(0,0))
        self.Gamma_label=ctk.CTkLabel(self.sidebar_frame,text="Flat hybridization constant \u0393:",anchor="w")
        self.Gamma_label.grid(row=9,column=0,padx=20,pady=(5,0))
        self.Gamma_Entry=ctk.CTkEntry(self.sidebar_frame,placeholder_text="0.3")
        self.Gamma_Entry.grid(row=10,column=0,padx=20,pady=(0,0))
        self.scaling_label=ctk.CTkLabel(self.sidebar_frame,text="No. of poles:",anchor="w")
        self.scaling_label.grid(row=11,column=0,padx=20,pady=(5,0))
        self.scaling_optionemenu=ctk.CTkOptionMenu(self.sidebar_frame,values=["2","3","4","5","6"])
        self.scaling_optionemenu.grid(row=12,column=0,padx=20,pady=(0,0))
        self.Tk_label=ctk.CTkLabel(self.sidebar_frame,text="Simulated Temperatures kbT:",anchor="w")
        self.Tk_label.grid(row=13,column=0,padx=20,pady=(5,0))
        self.Tk_Entry=ctk.CTkEntry(self.sidebar_frame,placeholder_text="[0]")
        self.Tk_Entry.grid(row=14,column=0,padx=20,pady=(0,0))
        self.sidebar_button_3=ctk.CTkButton(self.sidebar_frame,text="Submit Parameters",command=self.parainit)
        self.sidebar_button_3.grid(row=15,column=0,padx=20,pady=(15,15))
        self.N_Entry.insert(0,str(self.DEDargs[0]))
        self.U_Entry.insert(0,str(self.DEDargs[2]))
        self.Sigma_Entry.insert(0,str(self.DEDargs[3]))
        self.Ed_Entry.insert(0,str(self.DEDargs[4]))
        self.Gamma_Entry.insert(0,str(self.DEDargs[5]))
        self.scaling_optionemenu.set(str(self.DEDargs[1]))
        self.Tk_Entry.insert(0,str(self.DEDargs[11]))
        self.entry=ctk.CTkEntry(self,placeholder_text="C:\\")
        self.entry.grid(row=3,column=1,columnspan=2,padx=(20,0),pady=(20,0),sticky="nsew")
        self.main_button=ctk.CTkButton(self,text="Submit file to load",fg_color="transparent",width=160,border_width=2,text_color=("gray10","#DCE4EE"),command=self.fileloader)
        self.main_button.grid(row=3,column=3,padx=(20,20),pady=(20,0),sticky='w')
        self.entry_2=ctk.CTkEntry(self,placeholder_text="example.json")
        self.entry_2.grid(row=4,column=1,columnspan=2,padx=(20,0),pady=10,sticky="nsew")
        self.main_button_2=ctk.CTkButton(self,text="Submit save location",fg_color="transparent",width=160,border_width=2,text_color=("gray10","#DCE4EE"),command=lambda:savfilename(self.entry_2))
        self.main_button_2.grid(row=4,column=3,padx=(20,20),pady=10,sticky='w')
        self.plot_frame=ctk.CTkFrame(self,width=250,height=380)
        self.plot_frame.grid(row=0,column=1,columnspan=2,rowspan=1,padx=(20,0),pady=(20,10),sticky="nsew")
        self.slider_progressbar_frame=ctk.CTkFrame(self)
        self.slider_progressbar_frame.grid(row=1,column=1,columnspan=2,padx=(20,0),pady=(5,0),sticky="nsew")
        self.slider_progressbar_frame.grid_columnconfigure((0,1,2,3,4),weight=1)
        self.progressbar_1=ProgressBar(master=self.slider_progressbar_frame,variable=ctk.IntVar(value=0),itnum=self.pbar.n,Total=self.pbar.total,height=30)
        self.progressbar_1.grid(row=0,column=0,columnspan=5,padx=20,pady=(5,0),sticky="nsew")
        self.start_button=ctk.CTkButton(self.slider_progressbar_frame,text="Start",command=self.startDED)
        self.start_button.grid(row=1,column=0,padx=5,pady=(5,10))
        self.pause_button=ctk.CTkButton(self.slider_progressbar_frame,text="Pause",command=self.pauseDED)
        self.pause_button.grid(row=1,column=1,padx=5,pady=(5,10))
        self.stop_button=ctk.CTkButton(self.slider_progressbar_frame,text="Stop",command=self.stopDED)
        self.stop_button.grid(row=1,column=2,padx=5,pady=(5,10))
        self.show_button=ctk.CTkButton(self.slider_progressbar_frame,text="Show Results",command=self.showgraph)
        self.show_button.grid(row=1,column=3,padx=5,pady=(5,10))
        self.save_button=ctk.CTkButton(self.slider_progressbar_frame,text="Save Graph",command=lambda:savegraph(self,self.entry_2))
        self.save_button.grid(row=1,column=4,padx=5,pady=(5,10))
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.settings_tab=ctk.CTkTabview(self, width=80)
        self.settings_tab.grid(row=0,column=3,rowspan=2,padx=(20,20),pady=(20,0),sticky="nsew")
        self.settings_tab.add("Adv.")
        self.settings_tab.add("Multi orb.")
        self.settings_tab.tab("Adv.").grid_columnconfigure(0,weight=1)
        self.settings_tab.tab("Multi orb.").grid_columnconfigure(0,weight=1)
        self.scrollable_tab=ctk.CTkScrollableFrame(self.settings_tab.tab("Adv."),height=350,width=220,label_text="Advanced settings",label_font=ctk.CTkFont(size=20,weight="bold"),label_fg_color="transparent")
        self.scrollable_tab.grid(row=0, column=0,rowspan=2, padx=(0, 0), pady=(0, 0))
        self.eta_label=ctk.CTkLabel(self.scrollable_tab,text="Imaginary part of frequency arg. of\nGreen's function \u03B7 ([slope,offset]):",anchor="w")
        self.eta_label.grid(row=0,column=0,padx=10,pady=(5,0))
        self.eta_Entry=ctk.CTkEntry(self.scrollable_tab,placeholder_text="[0.02, 1e-39]")
        self.eta_Entry.grid(row=1,column=0,padx=10,pady=(2,0))
        self.eta_Entry.insert(0,str(self.DEDargs[12]))
        self.SizeO_label=ctk.CTkLabel(self.scrollable_tab,text="No. of energies in spectrum \u03C9:",anchor="w")
        self.SizeO_label.grid(row=2,column=0,padx=10,pady=(5,0))
        self.SizeO_Entry=ctk.CTkEntry(self.scrollable_tab,placeholder_text="1001")
        self.SizeO_Entry.grid(row=3,column=0,padx=10,pady=(0,0))
        self.SizeO_Entry.insert(0,str(self.DEDargs[13]))
        self.bound_label=ctk.CTkLabel(self.scrollable_tab,text="Range energies in spectrum \u03C9:",anchor="w")
        self.bound_label.grid(row=4,column=0,padx=10,pady=(5,0))
        self.bound_Entry=ctk.CTkEntry(self.scrollable_tab,placeholder_text="3")
        self.bound_Entry.grid(row=5,column=0,padx=10,pady=(0,0))
        self.bound_Entry.insert(0,str(self.DEDargs[14]))
        self.ctype_label=ctk.CTkLabel(self.scrollable_tab,text="Constraint type:",anchor="w")
        self.ctype_label.grid(row=6,column=0,padx=10,pady=(5,0))
        self.ctype_optionemenu=ctk.CTkOptionMenu(self.scrollable_tab,values=["n"," ","n%2","sn","ssn","dn","mosn"])
        self.ctype_optionemenu.grid(row=7,column=0,padx=10,pady=(0,0))
        self.log_checkbox=ctk.CTkCheckBox(master=self.scrollable_tab,text="Logarithmic scale energies \u03C9")
        self.log_checkbox.grid(row=8,column=0,padx=10,pady=(10,0))
        self.base_label=ctk.CTkLabel(self.scrollable_tab,text="Base log. scale:",anchor="w")
        self.base_label.grid(row=9,column=0,padx=10,pady=(5,0))
        self.base_Entry=ctk.CTkEntry(self.scrollable_tab,placeholder_text="1.5")
        self.base_Entry.grid(row=10,column=0,padx=10,pady=(0,0))
        self.base_Entry.insert(0,str(self.DEDargs[17]))
        self.Edcalc_label=ctk.CTkLabel(self.scrollable_tab,text="\u03B5d in interacting DOS:",anchor="w")
        self.Edcalc_label.grid(row=11,column=0,padx=10,pady=(5,0))
        self.Edcalc_optionemenu=ctk.CTkOptionMenu(self.scrollable_tab,values=["","AS"])
        self.Edcalc_optionemenu.grid(row=12,column=0,padx=10,pady=(0,0))
        self.tab_label2=ctk.CTkLabel(self.settings_tab.tab("Multi orb."),text="Multi orb. settings",font=ctk.CTkFont(size=20,weight="bold"))
        self.tab_label2.grid(row=0,column=0,padx=10,pady=(10,10))
        self.Nimpurities_label=ctk.CTkLabel(self.settings_tab.tab("Multi orb."),text="No. of orbitals:",anchor="w")
        self.Nimpurities_label.grid(row=1,column=0,padx=10,pady=(5,0))
        self.Nimpurities_optionemenu=ctk.CTkOptionMenu(self.settings_tab.tab("Multi orb."),values=["1","2"])
        self.Nimpurities_optionemenu.grid(row=2,column=0,padx=10,pady=(0,0))
        self.U2_label=ctk.CTkLabel(self.settings_tab.tab("Multi orb."),text="Inter-orbital Coulomb repulsion U':",anchor="w")
        self.U2_label.grid(row=3,column=0,padx=10,pady=(5,0))
        self.U2_Entry=ctk.CTkEntry(self.settings_tab.tab("Multi orb."),placeholder_text="0")
        self.U2_Entry.grid(row=4,column=0,padx=10,pady=(0,0))
        self.U2_Entry.insert(0,str(self.DEDargs[9]))
        self.J_label=ctk.CTkLabel(self.settings_tab.tab("Multi orb."),text="Hund’s rule coupling JH:",anchor="w")
        self.J_label.grid(row=5,column=0,padx=10,pady=(5,0))
        self.J_Entry=ctk.CTkEntry(self.settings_tab.tab("Multi orb."),placeholder_text="0")
        self.J_Entry.grid(row=6,column=0,padx=10,pady=(0,0))
        self.J_Entry.insert(0,str(self.DEDargs[10]))
        self.protocol('WM_DELETE_WINDOW',self.enableroot)

    def enableroot(self):
        self.msg=CTkMessagebox(master=self,title="Exit?",message="Do you want to close the program?\nUnsaved progress will be lost.",icon="question",option_1="Cancel",option_2="No",option_3="Yes")
        if self.msg.get()=="Yes":
            self.root.scaling_optionemenu.configure(state="normal")
            self.root.button_open.configure(state="normal")
            self.lift(self.root)
            self.destroy()

    def parainit(self):
        if not self.started:
            try:
                if self.pbar.n<int(self.N_Entry.get()):
                    self.pbar.total=self.DEDargs[0]=int(self.N_Entry.get())
                    self.N_Entry.configure(state="disabled")
                    self.start_button.configure(state="normal")
                    self.pbar.refresh()
                else:
                    self.N_Entry.delete(0,last_index=tk.END)
                    self.N_Entry.insert(0,str(self.pbar.n))
                self.DEDargs[1:],self.progressbar_1.Total,self.progressbar_1.itnum=[int(self.scaling_optionemenu.get()),float(self.U_Entry.get()),float(self.Sigma_Entry.get()),float(self.Ed_Entry.get()),float(self.Gamma_Entry.get()),self.ctype_optionemenu.get(),self.Edcalc_optionemenu.get(),int(self.Nimpurities_optionemenu.get()),float(self.U2_Entry.get()),float(self.J_Entry.get()),literal_eval(self.Tk_Entry.get()),literal_eval(self.eta_Entry.get()),int(self.SizeO_Entry.get()),float(self.bound_Entry.get()),self.DEDargs[15],bool(self.log_checkbox.get()),float(self.base_Entry.get())],self.DEDargs[0],self.pbar.n
                self.progressbar_1.set(self.pbar.n/self.DEDargs[0])
                self.U_Entry.configure(state="disabled")
                self.Sigma_Entry.configure(state="disabled")
                self.Ed_Entry.configure(state="disabled")
                self.Gamma_Entry.configure(state="disabled")
                self.scaling_optionemenu.configure(state="disabled")
                self.Tk_Entry.configure(state="disabled")
                self.sidebar_button_3.configure(state="disabled")
                self.eta_Entry.configure(state="disabled")
                self.SizeO_Entry.configure(state="disabled")
                self.bound_Entry.configure(state="disabled")
                self.ctype_optionemenu.configure(state="disabled")
                self.log_checkbox.configure(state="disabled")
                self.base_Entry.configure(state="disabled")
                self.Edcalc_optionemenu.configure(state="disabled")
                self.Nimpurities_optionemenu.configure(state="disabled")
                self.U2_Entry.configure(state="disabled")
                self.J_Entry.configure(state="disabled")
                if not self.loaded:
                    if self.DEDargs[16]: self.omega,self.Npoles=np.concatenate((-np.logspace(np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),np.log(1e-5)/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]),np.logspace(np.log(1e-5)/np.log(self.DEDargs[17]),np.log(self.DEDargs[14])/np.log(self.DEDargs[17]),int(np.round(self.DEDargs[13]/2)),base=self.DEDargs[17]))),int(self.DEDargs[1]/self.DEDargs[8])
                    else: self.omega,self.Npoles=np.linspace(-self.DEDargs[14],self.DEDargs[14],self.DEDargs[13]),int(self.DEDargs[1]/self.DEDargs[8])
                    self.eta,self.AvgSigmadat,self.Nfin,self.nd,self.Npoles,self.c,self.Lor=self.DEDargs[12][0]*abs(self.omega)+self.DEDargs[12][1],np.zeros((len(self.DEDargs[11]),self.DEDargs[13]),dtype='complex_'),np.zeros(len(self.DEDargs[11]),dtype='float'),np.zeros(len(self.DEDargs[11]),dtype='complex_'),int(self.DEDargs[1]/self.DEDargs[8]),[Jordan_wigner_transform(i,2*self.DEDargs[1]) for i in range(2*self.DEDargs[1])],Lorentzian(self.omega,self.DEDargs[5],self.DEDargs[1],self.DEDargs[4],self.DEDargs[3])[0]
                    (self.Hn,self.n)=Operators(self.c,self.DEDargs[8],self.DEDargs[1])
            except: pass

    def fileloader(self):
        if not self.started:
            try:
                self.data=AvgSigmajsonfileR(self.entry.get())
                self.Nfin,self.omega,self.AvgSigmadat,self.nd=np.array(self.data["Nfin"]),np.array(self.data["omega"]),np.array(self.data["AvgSigmadat"]*self.data["Nfin"]).squeeze(),np.array(np.array(self.data["nd"],dtype=np.complex128)*np.array(self.data["Nfin"],dtype=np.float64),dtype=np.complex128)
                self.DEDargs=[self.data["Ntot"],self.data["poles"],self.data["U"],self.data["Sigma"],self.data["Ed"],self.data["Gamma"],self.data["ctype"],self.data["Edcalc"],self.data["Nimpurities"],self.data["U2"],self.data["J"],self.data["Tk"],self.data["etaco"],self.data["SizeO"],self.data["bound"],self.data["posb"],self.data["log"],self.data["base"]]
                self.progressbar_1.Total,self.progressbar_1.itnum,self.telapsed=self.pbar.total,self.pbar.n=self.DEDargs[0],self.data["Nit"],self.data["telapsed"]
                self.progressbar_1.set(self.pbar.n/self.DEDargs[0])
                self.eta,self.Npoles,self.c,self.Lor=self.DEDargs[12][0]*abs(self.omega)+self.DEDargs[12][1],int(self.DEDargs[1]/self.DEDargs[8]),[Jordan_wigner_transform(i,2*self.DEDargs[1]) for i in range(2*self.DEDargs[1])],Lorentzian(self.omega,self.DEDargs[5],self.DEDargs[1],self.DEDargs[4],self.DEDargs[3])[0]
                (self.Hn,self.n),self.loaded=Operators(self.c,self.DEDargs[8],self.DEDargs[1]),True
                self.pbar.refresh()
                self.U_Entry.delete(0,last_index=tk.END)
                self.U_Entry.insert(0,str(self.DEDargs[2]))
                self.Sigma_Entry.delete(0,last_index=tk.END)
                self.Sigma_Entry.insert(0,str(self.DEDargs[3]))
                self.Ed_Entry.delete(0,last_index=tk.END)
                self.Ed_Entry.insert(0,str(self.DEDargs[4]))
                self.Gamma_Entry.delete(0,last_index=tk.END)
                self.Gamma_Entry.insert(0,str(self.DEDargs[5]))
                self.scaling_optionemenu.set(str(self.DEDargs[1]))
                self.Tk_Entry.delete(0,last_index=tk.END)
                self.Tk_Entry.insert(0,str(self.DEDargs[11]))
                self.eta_Entry.delete(0,last_index=tk.END)
                self.eta_Entry.insert(0,str(self.DEDargs[12]))
                self.SizeO_Entry.delete(0,last_index=tk.END)
                self.SizeO_Entry.insert(0,str(self.DEDargs[13]))
                self.bound_Entry.delete(0,last_index=tk.END)
                self.bound_Entry.insert(0,str(self.DEDargs[14]))
                self.ctype_optionemenu.set(str(self.DEDargs[6]))
                self.log_checkbox.configure(variable=tk.IntVar(value=int(self.DEDargs[16])))
                self.base_Entry.delete(0,last_index=tk.END)
                self.base_Entry.insert(0,str(self.DEDargs[17]))
                self.Edcalc_optionemenu.set(str(self.DEDargs[7]))
                self.Nimpurities_optionemenu.set(str(self.DEDargs[8]))
                self.U2_Entry.delete(0,last_index=tk.END)
                self.U2_Entry.insert(0,str(self.DEDargs[9]))
                self.J_Entry.delete(0,last_index=tk.END)
                self.J_Entry.insert(0,str(self.DEDargs[10]))
                if self.pbar.n>self.DEDargs[0]:
                    self.N_Entry.delete(0,last_index=tk.END)
                    self.N_Entry.insert(0,str(self.pbar.n))
                else:
                    self.N_Entry.delete(0,last_index=tk.END)
                    self.N_Entry.insert(0,str(self.DEDargs[0]))   
                if self.pbar.n==self.DEDargs[0]: self.start_button.configure(state="disabled")             
                self.U_Entry.configure(state="disabled")
                self.Sigma_Entry.configure(state="disabled")
                self.Ed_Entry.configure(state="disabled")
                self.Gamma_Entry.configure(state="disabled")
                self.scaling_optionemenu.configure(state="disabled")
                self.Tk_Entry.configure(state="disabled")
                self.eta_Entry.configure(state="disabled")
                self.SizeO_Entry.configure(state="disabled")
                self.bound_Entry.configure(state="disabled")
                self.ctype_optionemenu.configure(state="disabled")
                self.log_checkbox.configure(state="disabled")
                self.base_Entry.configure(state="disabled")
                self.Edcalc_optionemenu.configure(state="disabled")
                self.Nimpurities_optionemenu.configure(state="disabled")
                self.U2_Entry.configure(state="disabled")
                self.J_Entry.configure(state="disabled")
            except IOError or FileNotFoundError:
                self.entry.delete(0,last_index=tk.END)
                self.entry.insert(0,'Try again')
    
    def startDED(self):
        self.start_button.configure(state="disabled")
        self.started,self.stopped,self.pbar.start_t=True,False,self.pbar._time()-self.telapsed
        self.N_Entry.configure(state="disabled")
        self.U_Entry.configure(state="disabled")
        self.Sigma_Entry.configure(state="disabled")
        self.Ed_Entry.configure(state="disabled")
        self.Gamma_Entry.configure(state="disabled")
        self.scaling_optionemenu.configure(state="disabled")
        self.Tk_Entry.configure(state="disabled")
        self.sidebar_button_3.configure(state="disabled")
        self.eta_Entry.configure(state="disabled")
        self.SizeO_Entry.configure(state="disabled")
        self.bound_Entry.configure(state="disabled")
        self.ctype_optionemenu.configure(state="disabled")
        self.log_checkbox.configure(state="disabled")
        self.base_Entry.configure(state="disabled")
        self.Edcalc_optionemenu.configure(state="disabled")
        self.Nimpurities_optionemenu.configure(state="disabled")
        self.U2_Entry.configure(state="disabled")
        self.J_Entry.configure(state="disabled")
        self.loopDED()
        self.pause_button.configure(state="normal")
        self.stop_button.configure(state="normal")

    def loopDED(self):
        if not self.stopped and not self.paused and self.pbar.n!=self.DEDargs[0]:
            self.iterationDED()
            self.progressbar_1.itnum=self.pbar.n
            self.progressbar_1.set(self.pbar.n/self.DEDargs[0])
        if not self.stopped and self.pbar.n<self.DEDargs[0]:
            if self.DEDargs[1]>5:
                if self.pbar.n%100==0 and self.pbar.total>100:
                    _=savedata(self,self.entry_2)
                    self.showgraph()
                self.after(100,self.loopDED)
            else: self.after(1,self.loopDED)
        elif self.pbar.n>=self.DEDargs[0]: 
            self.pbar.close()
            self.stopDED()
            self.showgraph()
            savegraph(self,self.entry_2)

    def iterationDED(self,reset=False):
        while not reset:
            self.NewM,self.nonG,_=Startrans(self.Npoles,np.sort(Lorentzian(self.omega,self.DEDargs[5],self.Npoles,self.DEDargs[4],self.DEDargs[3])[1]),self.omega,self.eta)
            self.H0,self.H=HamiltonianAIM(np.repeat(self.NewM[0][0],self.DEDargs[8]),np.tile([self.NewM[k+1][k+1] for k in range(len(self.NewM)-1)],(self.DEDargs[8],1)),np.tile(self.NewM[0,1:],(self.DEDargs[8],1)),self.DEDargs[2],self.DEDargs[3],self.DEDargs[9],self.DEDargs[10],self.Hn)
            try: (self.MBGdat,self.Boltzmann,self.Ev0),reset=Constraint(self.DEDargs[6],self.H0,self.H,self.omega,self.eta,self.c,self.n,self.DEDargs[11],np.array([ar<self.DEDargs[0] for ar in self.Nfin]))
            except (np.linalg.LinAlgError,ValueError,scipy.sparse.linalg.ArpackNoConvergence): (self.MBGdat,self.Boltzmann,self.Ev0),reset=(np.zeros(len(self.omega),dtype='complex_'),np.zeros(len(self.DEDargs[11])),np.array([])),False
            if np.isnan(1/self.nonG-1/self.MBGdat+self.DEDargs[3]).any() or np.array([i>=1000 for i in np.real(1/self.nonG-1/self.MBGdat+self.DEDargs[3])]).any(): reset=False
        self.Nfin,self.AvgSigmadat,self.nd=self.Nfin+self.Boltzmann,self.AvgSigmadat+(1/self.nonG-1/self.MBGdat+self.DEDargs[3])*self.Boltzmann[:,None],self.nd+np.conj(self.Ev0).T@sum(self.Hn[0]).data.tocoo()@self.Ev0*self.Boltzmann
        if self.DEDargs[6]=='sn':self.pbar.n+=1
        else: self.pbar.n=int(min(self.Nfin))
        self.pbar.refresh()

    def pauseDED(self):
        if self.started: self.paused=not self.paused

    def stopDED(self):
        if savedata(self,self.entry_2) and not self.stopped and self.started:
            self.stopped,self.paused,self.started=True,False,False
            self.pause_button.configure(state="disabled")
            self.stop_button.configure(state="disabled")
            if self.pbar.n!=self.DEDargs[0]:
                self.start_button.configure(state="normal")

    def showgraph(self):
        mpl.rc('axes',edgecolor='white')
        if self.DEDargs[7]=='AS':self.fig,self.axis_font,self.fDOS=plt.figure(figsize=(9.5,7.6),dpi=50*self._get_window_scaling()),{'fontname':'Calibri','size':'19'},(-np.imag(np.nan_to_num(1/(self.omega-self.AvgSigmadat/self.Nfin[:,None]+(self.AvgSigmadat[:,int(np.round(self.DEDargs[13]/2))]/self.Nfin)[:,None]+1j*self.DEDargs[5])))/np.pi).squeeze()
        else:self.fig,self.axis_font,self.fDOS=plt.figure(figsize=(9.5,7.6),dpi=50*self._get_window_scaling()),{'fontname':'Calibri','size':'19'},(-np.imag(np.nan_to_num(1/(self.omega-self.AvgSigmadat/self.Nfin[:,None]-self.DEDargs[4]+1j*self.DEDargs[5])))/np.pi).squeeze()
        plt.rc('legend',fontsize=14)
        plt.rc('font',size=19)
        plt.rc('xtick',labelsize=17,color='white')
        plt.rc('ytick',labelsize=17,color='white')
        plt.xlim(min(self.omega),max(self.omega))
        plt.gca().set_ylim(bottom=0,top=1.2)
        plt.gca().set_xticks(np.linspace(min(self.omega),max(self.omega),2*int(max(self.omega))+1),minor=False)
        plt.xlabel("$\\omega$ [-]",**self.axis_font,color='white')
        plt.gca().set_ylabel("$\\rho$($\\omega$)",va="bottom",rotation=0,labelpad=30,**self.axis_font,color='white')
        plt.plot(self.omega,self.Lor,'--r',linewidth=4,label='$\\rho_0$')
        plt.plot(self.omega,self.fDOS,'-b',label='$\\rho,$n='+str(self.DEDargs[1]))
        plt.legend(fancybox=False).get_frame().set_edgecolor('black')
        plt.grid()
        plt.tight_layout()
        self.fig.set_facecolor("none")
        plt.gca().set_facecolor("#242424")
        self.plot_frame.configure(fg_color="transparent")
        self.canvas=FigureCanvasTkAgg(self.fig,master=self.plot_frame)
        self.canvas.get_tk_widget().config(bg="#242424")
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1,column=1)
        self.plot_frame.grid_rowconfigure((0,2),weight=1)
        self.plot_frame.grid_columnconfigure((0,2),weight=1)
        self.plot_frame.update()
        plt.close()

class polesWINDOW(ctk.CTkToplevel):
    def __init__(self,selfroot,N=200000,poles=4,U=3,Sigma=3/2,Ed=-3/2,Gamma=0.3,SizeO=1001,etaco=[0.02,1e-39],ctype='n',bound=3,Tk=[0],Nimpurities=1,U2=0,J=0,posb=1,log=False,base=1.5,*args,**kwargs):
        super().__init__(*args,**kwargs)

    pass

class ASAIMWINDOW(ctk.CTkToplevel):
    pass

class GNRWINDOW(ctk.CTkToplevel):
    pass

class EntropyWINDOW(ctk.CTkToplevel):
    pass

class StdevWINDOW(ctk.CTkToplevel):
    pass

if __name__ == "__main__":
    mainApp().mainloop()