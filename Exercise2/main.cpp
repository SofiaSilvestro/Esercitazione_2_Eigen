#include <iostream>
#include "Eigen/Eigen"

using namespace Eigen;
using namespace std;

// Function to solve the linear system Ax = b with PALU decomposition
bool SolveSystemPALU(const MatrixXd& A, const VectorXd& b,
                     const VectorXd& exactSolution, double& detA,
                     double& condA, double& errRelPALU)
{
    JacobiSVD<MatrixXd> svd(A);
    VectorXd singularValuesA = svd.singularValues();
    condA = singularValuesA.maxCoeff() / singularValuesA.minCoeff();
    detA = A.determinant();

    if(singularValuesA.minCoeff() < 1e-12)
    {
        errRelPALU = -1;
        return false;
    }

    VectorXd x = A.fullPivLu().solve(b);
    // Calculation the relative error
    errRelPALU = (exactSolution - x).norm() / exactSolution.norm();
    return true;
}

// Function to solve the linear system Ax = b with QR decomposition
VectorXd SolveSystemQR(const MatrixXd& A, const VectorXd& b,
                       const VectorXd& exactSolution, double& errRelQR)
{
    HouseholderQR<MatrixXd> qr(A);
    VectorXd x = qr.solve(b);
    // Calculation the relative error
    errRelQR = (exactSolution - x).norm() / exactSolution.norm();
    return x;
}


int main() {

    // Initialization of the exact solution
    VectorXd exactSolution = VectorXd::Ones(2);
    exactSolution << -1, -1;

    // Declaration and initialization of the FIRST matrix and its vector
    MatrixXd A1 = MatrixXd::Ones(2, 2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    VectorXd b1 = VectorXd::Ones(2);
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    double detA1;
    double condA1;
    double errRelPALU1;
    double errRelQR1;
    // Solution with PALU
    if (SolveSystemPALU(A1, b1, exactSolution, detA1, condA1, errRelPALU1))
        cout << scientific << "A1:\n" << A1 << "\nSolution with PALU decomposition:\n"
             << "detA1 = " << detA1 << ", RCondA1 = " << 1.0 / condA1
             << ", relative error = " << errRelPALU1 << endl;
    else
        cout << scientific << "A1:\n" << A1 << "\nSolution with PALU decomposition:\n"
             << "detA1 = " << detA1 << ", RCondA1 = " << 1.0 / condA1
             << "\nSingular matrix\n" << endl;
    // Solution with QR
    VectorXd x1 = SolveSystemQR(A1, b1, exactSolution, errRelQR1);
    cout << "Solution with QR decomposition:\n"
         << "Relative error: " << errRelQR1 << "\n" << endl;



    // Declaration and initialization of the SECOND matrix and its vector
    MatrixXd A2 = MatrixXd::Ones(2, 2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    VectorXd b2 = VectorXd::Ones(2);
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    double detA2;
    double condA2;
    double errRelPALU2;
    double errRelQR2;
    // Solution with PALU
    if (SolveSystemPALU(A2, b2, exactSolution,  detA2, condA2, errRelPALU2))
        cout << scientific << "A2:\n" << A2 << "\nSolution with PALU decomposition:\n"
             << "detA2 = " << detA2 << ", RCondA2 = " << 1.0 / condA2
             << ", relative error = " << errRelPALU2 << endl;
    else
        cout << scientific << "A2:\n" << A2 << "\nSolution with PALU decomposition:\n"
             << "detA2 = " << detA2 << ", RCondA2 = " << 1.0 / condA2
             << "\nSingular matrix\n" << endl;
    // Solution with QR
    VectorXd x2 = SolveSystemQR(A2, b2, exactSolution, errRelQR2);
    cout << "Solution with QR decomposition:\n"
         << "Relative error: " << errRelQR2 << "\n" << endl;




    // Declaration and initialization of the THIRD matrix and its vector
    MatrixXd A3 = MatrixXd::Ones(2, 2);
    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    VectorXd b3 = VectorXd::Ones(2);
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    double detA3;
    double condA3;
    double errRelPALU3;
    double errRelQR3;
    // Solution with PALU
    if (SolveSystemPALU(A3, b3, exactSolution, detA3, condA3, errRelPALU3))
        cout << scientific << "A3:\n" << A3 << "\nSolution with PALU decomposition:\n"
             << "detA3 = " << detA3 << ", RCondA3 = " << 1.0 / condA3
             << ", relative error = " << errRelPALU3 << endl;
    else
        cout << scientific << "A3:\n" << A3 << "\nSolution with PALU decomposition:\n"
             << "detA3 = " << detA3 << ", RCondA3 = " << 1.0 / condA3
             << "\nSingular matrix\n" << endl;
    // Solution with QR
    VectorXd x3 = SolveSystemQR(A3, b3, exactSolution, errRelQR3);
    cout << "Solution with QR decomposition:\n"
         << "Relative error: " << errRelQR3 << "\n" << endl;



    return 0;
}
