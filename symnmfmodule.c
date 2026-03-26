#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"
#include <stdlib.h>

/* Check if i should add print err msg from C in the header file and then here */
/* Check if using allocate matrix from C is allowed */
/* --- פונקציות עזר להמרה בין פייתון ל-C --- */

/* ממירה רשימה דו-ממדית של פייתון למטריצה ב-C */
static double **convert_py_to_c_mat(PyObject *py_mat, int rows, int cols)
{
    int i, j;
    PyObject *row, *item;

    double **c_mat = allocate_matrix(rows, cols);

    for (i = 0; i < rows; i++)
    {
        row = PyList_GetItem(py_mat, i); /* get python row */
        for (j = 0; j < cols; j++)
        {
            item = PyList_GetItem(row, j); /* get python item in (i,j) */
            c_mat[i][j] = PyFloat_AsDouble(item);
        }
    }
    return c_mat;
}

static PyObject *convert_c_to_py_mat(double **c_mat, int rows, int cols)
{
    int i, j;
    PyObject *py_mat = PyList_New(rows);
    PyObject *py_row;

    for (i = 0; i < rows; i++)
    {
        py_row = PyList_New(cols);
        for (j = 0; j < cols; j++)
        {
            PyList_SetItem(py_row, j, PyFloat_FromDouble(c_mat[i][j]));
        }
        PyList_SetItem(py_mat, i, py_row);
    }
    return py_mat;
}

/* --- פונקציות המעטפת (Wrappers) --- */

/* עטיפה עבור חישוב מטריצת Similarity */
static PyObject *sym_wrapper(PyObject *self, PyObject *args)
{
    PyObject *py_datapoints;
    int n, d;
    double **data_points, **sym_mat;
    PyObject *py_result;

    /* מקבלים מפייתון את המטריצה, מספר השורות ומספר העמודות */
    if (!PyArg_ParseTuple(args, "Oii", &py_datapoints, &n, &d))
    {
        return NULL;
    }

    /* 1. המרה ל-C */
    data_points = convert_py_to_c_mat(py_datapoints, n, d);

    /* 2. קריאה לפונקציה שלך */
    sym_mat = create_sym_matrix(data_points, n, d);

    /* 3. המרה חזרה לפייתון */
    py_result = convert_c_to_py_mat(sym_mat, n, n);

    /* 4. שחרור זיכרון C */
    free_matrix(data_points, n);
    free_matrix(sym_mat, n);

    return py_result;
}

/* עטיפה עבור חישוב SymNMF (אופטימיזציית H) */
static PyObject *symnmf_wrapper(PyObject *self, PyObject *args)
{
    PyObject *py_W, *py_H;
    int n, k;
    double **W, **H, **final_H;
    PyObject *py_result;

    /* מקבלים מפייתון את W, את H ההתחלתי, n ו-k */
    if (!PyArg_ParseTuple(args, "OOii", &py_W, &py_H, &n, &k))
    {
        return NULL;
    }

    W = convert_py_to_c_mat(py_W, n, n);
    H = convert_py_to_c_mat(py_H, n, k);

    /* קריאה לפונקציה שלך (היא משנה את H במקום, לכן final_H מצביע לאותו מקום כמו H) */
    final_H = optimize_H(W, H, n, k);

    py_result = convert_c_to_py_mat(final_H, n, k);

    free_matrix(W, n);
    free_matrix(H, n);

    return py_result;
}

/* עטיפה עבור חישוב מטריצת Diagonal Degree (ddg) */
static PyObject *ddg_wrapper(PyObject *self, PyObject *args)
{
    PyObject *py_datapoints;
    int n, d, i, j;
    double **data_points, **sym_mat;
    double *diag_arr;
    PyObject *py_result, *py_row;

    if (!PyArg_ParseTuple(args, "Oii", &py_datapoints, &n, &d))
    {
        return NULL;
    }

    data_points = convert_py_to_c_mat(py_datapoints, n, d);
    sym_mat = create_sym_matrix(data_points, n, d);
    diag_arr = create_diag_matrix(sym_mat, n);

    /* בניית המטריצה האלכסונית כרשימה דו-ממדית עבור פייתון */
    py_result = PyList_New(n);
    for (i = 0; i < n; i++)
    {
        py_row = PyList_New(n);
        for (j = 0; j < n; j++)
        {
            if (i == j)
            {
                PyList_SetItem(py_row, j, PyFloat_FromDouble(diag_arr[i]));
            }
            else
            {
                PyList_SetItem(py_row, j, PyFloat_FromDouble(0.0));
            }
        }
        PyList_SetItem(py_result, i, py_row);
    }

    free_matrix(data_points, n);
    free_matrix(sym_mat, n);
    free(diag_arr);

    return py_result;
}

/* עטיפה עבור חישוב מטריצת Normalized Similarity (norm) */
static PyObject *norm_wrapper(PyObject *self, PyObject *args)
{
    PyObject *py_datapoints;
    int n, d;
    double **data_points, **sym_mat, **norm_mat;
    double *diag_arr;
    PyObject *py_result;

    if (!PyArg_ParseTuple(args, "Oii", &py_datapoints, &n, &d))
    {
        return NULL;
    }

    data_points = convert_py_to_c_mat(py_datapoints, n, d);
    sym_mat = create_sym_matrix(data_points, n, d);
    diag_arr = create_diag_matrix(sym_mat, n);
    norm_mat = create_norm_matrix(sym_mat, diag_arr, n);

    py_result = convert_c_to_py_mat(norm_mat, n, n);

    free_matrix(data_points, n);
    free_matrix(sym_mat, n);
    free(diag_arr);
    free_matrix(norm_mat, n);

    return py_result;
}

/* צריך להוסיף כאן גם את ddg_wrapper ואת norm_wrapper על אותו עיקרון... */

/* --- הגדרת המודול עבור פייתון --- */

static PyMethodDef symnmfMethods[] = {
    {"sym",
     (PyCFunction)sym_wrapper,
     METH_VARARGS,
     "Calculates the similarity matrix"},

    {"symnmf",
     (PyCFunction)symnmf_wrapper,
     METH_VARARGS,
     "Optimizes H matrix"},

    {"ddg",
     (PyCFunction)ddg_wrapper,
     METH_VARARGS,
     "Calculates the Diagonal Degree Matrix"},

    {"norm",
     (PyCFunction)norm_wrapper,
     METH_VARARGS,
     "Calculates the normalized similarity matrix"},

    {NULL, NULL, 0, NULL} /* חייבים לסיים את הרשימה ככה */
};

/* הגדרת המודול עצמו */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf", /* שם המודול */
    NULL,     /* תיעוד המודול */
    -1,
    symnmfMethods};

/* פונקציית האתחול שפייתון מחפש כשהוא עושה import */
PyMODINIT_FUNC PyInit_symnmf(void)
{
    return PyModule_Create(&symnmfmodule);
}