import plotly.graph_objects as go
import torch

class LV_Plotter:
    def __init__(self, t, X, t_d, X_d, true_res_dx, true_res_dy):
        self.t = t
        self.X = X
        self.z_d = t_d
        self.U_d = X_d
        self.true_res_dx = true_res_dx
        self.true_res_dy = true_res_dy

    def __call__(self, u, G):
        device = next(u.parameters()).device  # Get the device of the model
        plots = {}
        set_free = lambda x: x.squeeze().cpu().detach().numpy()

        # Evaluate the model
        X_pred = u(self.t.unsqueeze(-1).to(device))

        # Plot with Plotly
        fig = go.Figure()
        for i, color, label in zip([0, 1], ['green', 'red'], ['Prey', 'Predator']):
            fig.add_scatter(x=set_free(self.t), y=set_free(self.X[:, i]), mode='lines', name=f'{label}', line=dict(dash='dash', color=color))
            fig.add_scatter(x=set_free(self.t), y=set_free(X_pred[:, i]), mode='lines', name=f'{label} (pred)', line=dict(color=color))
            fig.add_scatter(x=set_free(self.z_d), y=set_free(self.U_d[:, i]), mode='markers', name=f'{label} (data)', marker=dict(color=color, symbol='x'))
        
        fig.update_layout(title="Lotka-Volterra Model")
        plots["Solution"] = fig

        # Plot missing terms
        res = G(X_pred).cpu()

        fig = go.Figure()
        for res_val, label, color in zip([res[:, 0], res[:, 1]], ['Residual Prey', 'Residual Predator'], ['green', 'red']):
            fig.add_scatter(x=set_free(self.t), y=set_free(res_val), mode='lines', name=label, line=dict(color=color))
        fig.add_scatter(x=set_free(self.t), y=self.true_res_dx, mode='lines', name='Missing Prey', line=dict(dash='dash', color='green'))
        fig.add_scatter(x=set_free(self.t), y=self.true_res_dy, mode='lines', name='Missing Predator', line=dict(dash='dash', color='red'))
        fig.update_layout(title="Lotka-Volterra Missing Terms")

        plots["Missing Terms"] = fig
        return plots