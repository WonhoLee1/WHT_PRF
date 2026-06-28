import numpy as np

def convert_prony_to_degenerate_prf(g_i, tau_i, E_inst):
    E_i = g_i * E_inst
    A_i = 1.0 / (E_i * tau_i)
    return A_i

def main():
    print("=== Spreadsheet Verification ===")
    C10 = 0.152
    E_inst = 6 * C10
    
    # Using the exact rounded tau values from the image
    networks = [(0.0493, 1.3), (0.0541, 23.8), (0.0635, 497.5)]
    print("\n[Using values exactly as written in the image (rounded)]")
    for i, (g, tau) in enumerate(networks):
        A = convert_prony_to_degenerate_prf(g, tau, E_inst)
        q0 = 1.0 / A
        print(f"Network {i+1}: SR={g:.4f}, A={A:.4e}, q0={q0:.4f}")

    # The spreadsheet actually used unrounded tau values internally
    # We can reverse-engineer them from the spreadsheet's A values
    print("\n[Using unrounded tau values (Reverse-engineered from spreadsheet A)]")
    A_spreadsheet = [17.458, 0.85267, 0.034709]
    for i, (g, tau_rounded) in enumerate(networks):
        # Reverse engineer tau: A = 1 / (6 * g * tau * C10) => tau = 1 / (6 * g * C10 * A)
        tau_real = 1.0 / (6 * g * C10 * A_spreadsheet[i])
        A = convert_prony_to_degenerate_prf(g, tau_real, E_inst)
        q0 = 1.0 / A
        print(f"Network {i+1}: SR={g:.4f}, A={A:.4e} (Spreadsheet A={A_spreadsheet[i]:.4e}), q0={q0:.4f}, actual_tau={tau_real:.4f}")

if __name__ == "__main__":
    main()
