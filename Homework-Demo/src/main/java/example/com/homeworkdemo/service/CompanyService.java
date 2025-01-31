package example.com.homeworkdemo.service;

import example.com.homeworkdemo.model.Company;
import example.com.homeworkdemo.model.Transaction;

import java.util.List;

public interface CompanyService {
    List<Company> getAll();

    Company getByCompanyCode(String code);

    List<Transaction> getTransactionByCompanyCode(String code);

    List<String> getAllCodes();

    void fetchAndSaveCompany();

    void fetchAndSaveTransaction();
}