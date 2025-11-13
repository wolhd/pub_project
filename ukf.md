
Yes, you can absolutely use an **Earth-Centered, Earth-Fixed (ECEF)** coordinate system for your state vector when implementing a Kalman filter with range and azimuth measurements.

Since both the system dynamics (state propagation) and the measurement model (range and azimuth) will be **nonlinear** in ECEF, you'll need to use a variation like the **Extended Kalman Filter (EKF)** or the **Unscented Kalman Filter (UKF)**.

Here is a general guide on how to implement this:

## 🧭 Extended Kalman Filter (EKF) Implementation

The EKF is typically preferred for this type of problem because it handles nonlinear system and measurement models by using **linearization** (Jacobian matrices).

----------

### 1. State Vector Definition

Your state vector $\mathbf{x}_k$ at time $k$ will be the target's position and velocity in ECEF coordinates:

$$\mathbf{x}_k = [x_{ECEF}, y_{ECEF}, z_{ECEF}, \dot{x}_{ECEF}, \dot{y}_{ECEF}, \dot{z}_{ECEF}]^T$$

The system noise covariance $\mathbf{Q}_k$ will represent the uncertainty in your state propagation model (e.g., due to unmodeled accelerations).

----------

### 2. State Propagation (Prediction Step)

The prediction step projects the current state and its uncertainty forward in time using a dynamic model $f$.

-   Prediction:
    
    $$\hat{\mathbf{x}}_k^- = f(\hat{\mathbf{x}}_{k-1}^+, \mathbf{u}_k)$$
    
    The function $f$ represents your target's motion model (e.g., constant velocity, constant acceleration, or a specific orbital model). In ECEF, the rotation of the Earth must be accounted for in the dynamics if your model is not purely inertial.
    
-   Covariance Prediction:
    
    $$\mathbf{P}_k^- = \mathbf{F}_{k-1} \mathbf{P}_{k-1}^+ \mathbf{F}_{k-1}^T + \mathbf{Q}_k$$
    
    Where $\mathbf{F}_{k-1}$ is the State Transition Matrix (the Jacobian of $f$ with respect to the state $\mathbf{x}$), evaluated at the previous estimated state.
    

----------

### 3. Measurement Model (Update Step)

This step incorporates the range ($\rho$) and azimuth ($\alpha$) measurements.

#### A. Measurement Vector

Your measurement vector $\mathbf{z}_k$ is:

$$\mathbf{z}_k = [\rho, \alpha]^T$$

#### B. The Measurement Function $h(\mathbf{x})$

This function relates the predicted state $\hat{\mathbf{x}}_k^-$ (ECEF position) to the expected measurements in range and azimuth. This is where the nonlinearity lies.

First, you need the **sensor's ECEF position** $\mathbf{p}_{\text{sensor}} = [x_s, y_s, z_s]^T$.

The relative position of the target with respect to the sensor in ECEF is:

$$\mathbf{r}_{\text{ECEF}} = [x_{\text{ECEF}} - x_s, y_{\text{ECEF}} - y_s, z_{\text{ECEF}} - z_s]^T$$

The predicted range $\hat{\rho}$ is:

$$\hat{\rho} = \sqrt{(x_{\text{ECEF}} - x_s)^2 + (y_{\text{ECEF}} - y_s)^2 + (z_{\text{ECEF}} - z_s)^2}$$

To calculate the predicted azimuth $\hat{\alpha}$, you must transform the relative ECEF vector $\mathbf{r}_{\text{ECEF}}$ into a Local Tangent (ENU/NED) Coordinate System at the sensor location. This transformation involves the sensor's geodetic latitude $\phi_s$ and longitude $\lambda_s$.

$$\mathbf{r}_{\text{ENU}} = \mathbf{T}_{\text{ECEF} \to \text{ENU}}(\phi_s, \lambda_s) \cdot \mathbf{r}_{\text{ECEF}}$$

From the resulting East ($E$) and North ($N$) components in the ENU frame:

$$\hat{\alpha} = \arctan2(E, N)$$

The full measurement function is $\mathbf{h}(\hat{\mathbf{x}}_k^-) = [\hat{\rho}, \hat{\alpha}]^T$.

#### C. The Measurement Matrix $\mathbf{H}_k$

The measurement matrix $\mathbf{H}_k$ is the Jacobian of $\mathbf{h}(\mathbf{x})$ with respect to the state vector $\mathbf{x}$, evaluated at the predicted state $\hat{\mathbf{x}}_k^-$.

$$\mathbf{H}_k = \frac{\partial \mathbf{h}}{\partial \mathbf{x}}\bigg|_{\mathbf{x} = \hat{\mathbf{x}}_k^-}$$

This matrix calculation will be complex as it involves the partial derivatives of the range and the azimuth conversion (which includes the ECEF to ENU transformation) with respect to the six state variables ($x, y, z, \dot{x}, \dot{y}, \dot{z}$).

----------

### 4. Update Equations (Correction Step)

The rest of the EKF update follows the standard linear Kalman filter steps using the calculated $\mathbf{H}_k$ and the measurement noise covariance $\mathbf{R}_k$:

1.  Innovation (Measurement Residual):
    
    $$\mathbf{y}_k = \mathbf{z}_k - \mathbf{h}(\hat{\mathbf{x}}_k^-)$$
    
2.  Innovation Covariance:
    
    $$\mathbf{S}_k = \mathbf{H}_k \mathbf{P}_k^- \mathbf{H}_k^T + \mathbf{R}_k$$
    
3.  Kalman Gain:
    
    $$\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}_k^T \mathbf{S}_k^{-1}$$
    
4.  State Update:
    
    $$\hat{\mathbf{x}}_k^+ = \hat{\mathbf{x}}_k^- + \mathbf{K}_k \mathbf{y}_k$$
    
5.  Covariance Update:
    
    $$\mathbf{P}_k^+ = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_k^-$$
    

----------

## 💡 Alternatives

Due to the complex and potentially error-prone calculation of the Jacobian $\mathbf{H}_k$, the **Unscented Kalman Filter (UKF)** is often a more robust choice. The UKF avoids explicit linearization by using a set of deterministically chosen sample points (sigma points) to capture the nonlinearity of the transformation, leading to a generally more accurate estimate without the need to derive and code the Jacobian matrices.

Would you like a more detailed breakdown of the Jacobian matrix $\mathbf{H}_k$ for the EKF, or an overview of the UKF approach?







That's a great choice. The **Unscented Kalman Filter (UKF)** is generally more robust and accurate than the Extended Kalman Filter (EKF) for highly nonlinear problems, such as using range and azimuth measurements in the ECEF coordinate system, because it avoids the need to explicitly calculate complex Jacobian matrices.

The UKF achieves this by using a technique called the **Unscented Transformation (UT)**, which involves deterministically sampling points (called **sigma points**) around the current state estimate.

Here is how you would implement the UKF for your ECEF state and range/azimuth measurements:

----------

## 🔬 Unscented Kalman Filter (UKF) Implementation Steps

### 1. State Definition

Your state vector $\mathbf{x}$ remains the same as for the EKF: position and velocity in ECEF.

$$\mathbf{x}_k = [x_{ECEF}, y_{ECEF}, z_{ECEF}, \dot{x}_{ECEF}, \dot{y}_{ECEF}, \dot{z}_{ECEF}]^T$$

The dimension of the state is $n_x = 6$. You will need to calculate $2n_x + 1 = 13$ sigma points.

### 2. The Unscented Transformation (UT)

The UT is the core of the UKF. It approximates a probability distribution by using a minimal set of sigma points that capture the mean and covariance of the Gaussian.

#### A. Generating Sigma Points ($\mathcal{X}$)

At the beginning of each step (using the prior $\hat{\mathbf{x}}_{k-1}^+$ and $\mathbf{P}_{k-1}^+$), you generate the sigma points $\mathcal{X}_{k-1}$ and their associated weights ($\mathbf{w}^m$ for mean and $\mathbf{w}^c$ for covariance).

-   Sigma Points:
    
    $$\mathcal{X}_{k-1}^{(0)} = \hat{\mathbf{x}}_{k-1}^+$$
    
    $$\mathcal{X}_{k-1}^{(i)} = \hat{\mathbf{x}}_{k-1}^+ + (\sqrt{(n_x+\lambda)\mathbf{P}_{k-1}^+})_i \quad \text{for } i=1, \dots, n_x$$
    
    $$\mathcal{X}_{k-1}^{(i+n_x)} = \hat{\mathbf{x}}_{k-1}^+ - (\sqrt{(n_x+\lambda)\mathbf{P}_{k-1}^+})_i \quad \text{for } i=1, \dots, n_x$$
    
    Where $\lambda = \alpha^2(n_x + \kappa) - n_x$ is a scaling parameter, and $(\sqrt{\mathbf{P}})_i$ is the $i$-th column of the matrix square root (e.g., using Cholesky decomposition).
    

#### B. Calculating Weights ($\mathbf{w}$)

The weights depend on the tuning parameters $\alpha$ (controls spread of points), $\beta$ (incorporates prior knowledge of the distribution), and $\kappa$. A common choice is $\alpha=10^{-3}$, $\beta=2$, and $\kappa=0$ or $\kappa=3-n_x$.

### 3. Prediction Step

The sigma points are propagated directly through the nonlinear process model $f$ without linearization.

-   Propagate Sigma Points:
    
    $$\mathcal{Y}_k^{(i)} = f(\mathcal{X}_{k-1}^{(i)})$$
    
-   Predict Mean ($\hat{\mathbf{x}}_k^-$):
    
    The predicted state mean is the weighted average of the propagated sigma points.
    
    $$\hat{\mathbf{x}}_k^- = \sum_{i=0}^{2n_x} \mathbf{w}_i^m \mathcal{Y}_k^{(i)}$$
    
-   Predict Covariance ($\mathbf{P}_k^-$):
    
    The predicted covariance is the weighted covariance of the propagated sigma points, plus the process noise covariance $\mathbf{Q}_k$.
    
    $$\mathbf{P}_k^- = \sum_{i=0}^{2n_x} \mathbf{w}_i^c (\mathcal{Y}_k^{(i)} - \hat{\mathbf{x}}_k^-) (\mathcal{Y}_k^{(i)} - \hat{\mathbf{x}}_k^-)^T + \mathbf{Q}_k$$
    

----------

### 4. Update Step (Measurement Incorporation)

This step incorporates the range ($\rho$) and azimuth ($\alpha$) measurements, again using the unscented transformation.

#### A. Propagate Sigma Points through Measurement Function

The predicted sigma points $\mathcal{Y}_k$ are passed through the nonlinear measurement function $\mathbf{h}(\mathbf{x})$, which converts ECEF position to Range and Azimuth ($\mathbf{z} = [\rho, \alpha]^T$) as described in the EKF section.

-   Measurement Function:
    
    $$\mathcal{Z}_k^{(i)} = \mathbf{h}(\mathcal{Y}_k^{(i)})$$
    
    (This is where the complex ECEF to ENU to Range/Azimuth conversion is performed for each point).
    

#### B. Calculate Predicted Measurement Mean and Covariance

-   Predicted Measurement Mean ($\hat{\mathbf{z}}_k$):
    
    $$\hat{\mathbf{z}}_k = \sum_{i=0}^{2n_x} \mathbf{w}_i^m \mathcal{Z}_k^{(i)}$$
    
-   Innovation Covariance ($\mathbf{S}_k$):
    
    This is the covariance of the predicted measurements, plus the measurement noise covariance $\mathbf{R}_k$.
    
    $$\mathbf{S}_k = \sum_{i=0}^{2n_x} \mathbf{w}_i^c (\mathcal{Z}_k^{(i)} - \hat{\mathbf{z}}_k) (\mathcal{Z}_k^{(i)} - \hat{\mathbf{z}}_k)^T + \mathbf{R}_k$$
    

#### C. Calculate Cross-Covariance ($\mathbf{P}_{xz}$)

This matrix relates the state prediction errors to the measurement prediction errors.

$$\mathbf{P}_{xz} = \sum_{i=0}^{2n_x} \mathbf{w}_i^c (\mathcal{Y}_k^{(i)} - \hat{\mathbf{x}}_k^-) (\mathcal{Z}_k^{(i)} - \hat{\mathbf{z}}_k)^T$$

#### D. Calculate Kalman Gain and Update

-   Kalman Gain ($\mathbf{K}_k$):
    
    $$\mathbf{K}_k = \mathbf{P}_{xz} \mathbf{S}_k^{-1}$$
    
-   State Update ($\hat{\mathbf{x}}_k^+$):
    
    $$\hat{\mathbf{x}}_k^+ = \hat{\mathbf{x}}_k^- + \mathbf{K}_k (\mathbf{z}_k - \hat{\mathbf{z}}_k)$$
    
-   Covariance Update ($\mathbf{P}_k^+$):
    
    $$\mathbf{P}_k^+ = \mathbf{P}_k^- - \mathbf{K}_k \mathbf{S}_k \mathbf{K}_k^T$$
    

The key advantage here is that you replace the challenging **Jacobian** derivations with the straightforward, but computationally heavier, process of propagating the **sigma points** through the true nonlinear functions. This typically results in a more stable and accurate filter for this type of navigation problem.
