import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import datetime as dt

alpha = 1 / 5.8
gamma = 1 / 5
eff = 0.66
N = 70000000
date_to_index = {}

def plot_cases(df, P, start_date, end_date):
    plt.figure(figsize=(8, 8))
    
    dates = ['2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01', '2021-09-01']
    xtick_labels = ['April', 'May', 'June', 'July', 'August', 'September']
    plt.xticks([(date_to_index[i] - date_to_index[start_date])for i in dates], xtick_labels)
    
    P1 = np.array(P)

    y = SEIRV(df, start_date, end_date, P1)[:, 4]
    plt.plot(y, label = "Open Loop Control: Beta")
    
    P1[0] = 2 * P[0] / 3
    y = SEIRV(df, start_date, end_date, P1)[:, 4]
    plt.plot(y, label = "Open Loop Control: 2Beta/3")
    
    P1[0] = P[0] / 2
    y = SEIRV(df, start_date, end_date, P1)[:, 4]
    plt.plot(y, label = "Open Loop Control: Beta/2")

    P1[0] = P[0] / 3
    y = SEIRV(df, start_date, end_date, P1)[:, 4]
    plt.plot(y, label = "Open Loop Control: Beta/3")

    y = SEIRV(df, start_date, end_date, P, is_close_loop_control=True)[:, 4]
    plt.plot(y, label = "Close Loop Control")

    y = np.array(df.iloc[date_to_index[start_date] - 7: date_to_index[end_date]]['Confirmed'])
    for i in range(y.shape[0] - 1, 6, -1):
        y[i] = (y[i] - y[i-7]) / 7
    
    plt.plot(y[7:], label = "Actual Cases")
    
    plt.legend()
    plt.title('Open-Loop And Closed-Loop Control Predictions')
    plt.xlabel(f'{start_date} to {end_date}')
    plt.ylabel('#Cases Per Day')
    plt.savefig("cases_prediction.png")  
    
def plot_susceptible(df, P, start_date, end_date):
    plt.figure(figsize=(8, 8))
    
    dates = ['2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01', '2021-08-01', '2021-09-01']
    xtick_labels = ['April', 'May', 'June', 'July', 'August', 'September']
    plt.xticks([(date_to_index[i] - date_to_index[start_date])for i in dates], xtick_labels)
    
    P1 = np.array(P)

    y = SEIRV(df, start_date, end_date, P1)[:, 0] / N
    plt.plot(y, label = "Open Loop Control: Beta")
    
    P1[0] = 2 * P[0] / 3
    y = SEIRV(df, start_date, end_date, P1)[:, 0] / N
    plt.plot(y, label = "Open Loop Control: 2Beta/3")
    
    P1[0] = P[0] / 2
    y = SEIRV(df, start_date, end_date, P1)[:, 0] / N
    plt.plot(y, label = "Open Loop Control: Beta/2")

    P1[0] = P[0] / 3
    y = SEIRV(df, start_date, end_date, P1)[:, 0] / N
    plt.plot(y, label = "Open Loop Control: Beta/3")

    y = SEIRV(df, start_date, end_date, P, is_close_loop_control=True)[:, 0] / N
    plt.plot(y, label = "Close Loop Control")

    plt.legend()
    plt.title('Susceptible Fractions')
    plt.xlabel(f'{start_date} to {end_date}')
    plt.ylabel('#Susceptible / 70000000')
    plt.savefig("susceptible_prediction.png")

def read_file(filename):
    df = pd.read_csv(filename)
    df = df[['Date', 'Confirmed', 'Tested', 'First Dose Administered']]
    df.rename(columns = {"First Dose Administered": "Dose1"}, inplace = True)
    for i, date in enumerate(df['Date'].values.tolist()):
        date_to_index[date] = i
    return df

def deltaV(df, i, date):
    if date >= '2021-04-27':
        return 200000
    else:
        return df.iloc[i]["Dose1"] - df.iloc[i-1]["Dose1"]

def isTuesday(date, day_no):
    if day_no < 8:
        return False
    year, month, day = list(map(int, date.split('-')))
    return dt.datetime(year, month, day).strftime('%A') == 'Tuesday'

def get_new_beta(lst, Beta):
    avg_cases = 0
    beta = -1

    for j in range(-7, 0):
        avg_cases += lst[j][4]
    avg_cases /= 7

    if avg_cases < 10001:
        beta = Beta
    elif avg_cases < 25001:
        beta = 2 * Beta / 3
    elif avg_cases < 100001:
        beta = Beta / 2
    else:
        beta = Beta / 3

    return beta


def SEIRV(df, start_date, end_date, P, is_close_loop_control = False):
    Beta, S, E, I, R, CIR_0 = P

    start = date_to_index[start_date]
    end = date_to_index[end_date]
    lst = []

    R_0 = R
    beta = Beta
    T_0 = (df.iloc[start - 1]['Tested'] - df.iloc[start - 8]['Tested']) / 7

    for i in range(start, end + 1):
        date_i = df.iloc[i]['Date']

        if is_close_loop_control and isTuesday(date_i, i - start):
            beta = get_new_beta(lst, Beta)
        
        dW = 0
        if date_i <= '2021-04-15':
            dW = R_0 / 30
        elif date_i <= '2021-09-11':
            dW = 0
        else:
            dW = lst[-180][3] + eff * deltaV(df, i - 180, date_i)


        dV = deltaV(df, i, date_i)

        dS = (- beta * S * I / N) - (eff * dV) + dW
        dE = (beta * S * I / N) - (alpha * E)
        dI =  (alpha * E) - (gamma * I)
        dR = (gamma * I) + (eff * dV) - dW
        
        S = S + dS 
        E = E + dE
        I = I + dI
        R = R + dR

        T = (df.iloc[i - 1]['Tested'] - df.iloc[i - 8]['Tested']) / 7
        CIR = CIR_0 * T_0 / T
        cases_i = alpha * E / CIR

        lst.append([S, E, I, R, cases_i])

    return np.array(lst)

def compute_loss(df, start_date, end_date, estimated_SEIR, CIR_0):
    
    start = date_to_index[start_date]
    end = date_to_index[end_date]
    
    n = end - start + 1
    actual_c = np.zeros(n)
    estimated_c = np.zeros(n)
    
    T_0 = (df.iloc[start - 1]['Tested'] - df.iloc[start - 8]['Tested']) / 7
    
    for i in range(n):
        index = start + i
        T = (df.iloc[index - 1]['Tested'] - df.iloc[index - 8]['Tested']) / 7
        CIR = CIR_0 * T_0 / T
        
        actual_c[i] = df.iloc[index]['Confirmed'] - df.iloc[index - 1]['Confirmed']
        estimated_c[i] = alpha * estimated_SEIR[i][1] / CIR
    
    avg_actual_c = np.cumsum(actual_c)
    avg_estimated_c = np.cumsum(estimated_c)
        
    for i in range(n - 1, 6, -1):
        avg_actual_c[i] = (avg_actual_c[i] - avg_actual_c[i - 7]) / 7
        avg_estimated_c[i] = (avg_estimated_c[i] - avg_estimated_c[i - 7]) / 7

    for i in range(7):
        avg_actual_c[i] = avg_actual_c[i] / (i + 1)
        avg_estimated_c[i] = avg_estimated_c[i] / (i + 1)

    loss = np.sum((np.log(avg_actual_c) - np.log(avg_estimated_c)) ** 2) / n
    
    return loss

def gradient_descent(df, start_date, end_date, P):
    # P = [beta, S, E, I, R, CIR]
    iterations = 0
    step = np.array([0.01, 1, 1, 1, 1, 0.1])
    while iterations <= 10000:
        new_P = np.zeros(6)

        # computing the SEIR estimate for given parameters
        estimated_SEIR = SEIRV(df, start_date, end_date, P)
        loss_before = compute_loss(df, start_date, end_date, estimated_SEIR, P[5])
        if iterations % 5 == 0:
            print(f"#Iterations = {iterations}, Training Loss = {loss_before}")

        if loss_before < 0.01:
            break
        
        # computing the gradient w.r.t. to each parameter and the updating the parameters
        for i in range(6):
            P[i] += step[i]
            # computing the SEIR estimate after updating the ith parameter
            estimated_SEIR = SEIRV(df, start_date, end_date, P)
            loss_after = compute_loss(df, start_date, end_date, estimated_SEIR, P[5])
            P[i] -= step[i]
            # updating the parameter
            new_P[i] = P[i] - (loss_after - loss_before) / ((iterations + 1) * step[i])
            if new_P[i] < 0:
                new_P[i] = P[i]
        P = new_P
        iterations += 1       
    return P, loss_before

def main():
    df = read_file('COVID19_data.csv')
    
    beta = 4.5
    S_0 = 48999999.9
    E_0 = 73499.9180
    I_0 = 73499.9183
    R_0 = 20852999.9
    CIR_0 = 12
    training_start_date = '2021-03-16'
    training_end_date = '2021-04-26'
    
    P = np.array([beta, S_0, E_0, I_0, R_0, CIR_0])
    
    print("\n========================== Initial Parameters =======================")
    print(f'S(0) = {P[1]}, E(0) = {P[2]}, I(0) = {P[3]}, R(0) = {P[4]}')
    print(f'Beta = {P[0]}, CIR(0) = {P[5]}\n')
    
    opt_P, loss = gradient_descent(df, training_start_date, training_end_date, P)
    
    print("\n========================== Optimal Parameters =======================")
    print(f'S(0) = {opt_P[1]}, E(0) = {opt_P[2]}, I(0) = {opt_P[3]}, R(0) = {opt_P[4]}')
    print(f'Beta = {opt_P[0]}, CIR(0) = {opt_P[5]}')
    
    print(f"\nTraining Loss = {loss}")
    
    
    plot_cases(df, opt_P, '2021-03-16', '2021-09-19')
    plot_susceptible(df, opt_P, '2021-03-16', '2021-09-19')

if __name__ == '__main__':
    main()