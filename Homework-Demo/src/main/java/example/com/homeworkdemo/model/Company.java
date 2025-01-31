package example.com.homeworkdemo.model;

import com.fasterxml.jackson.annotation.JsonManagedReference;
import jakarta.persistence.*;

import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "Company")
public class Company {
    @Id
    private String companyCode;

    @OneToMany(mappedBy = "company")
    @JsonManagedReference
    List<Transaction> transactions;

    public Company(String companyCode) {
        this.companyCode = companyCode;
        transactions = new ArrayList<>();
    }

    public Company() {}

    public String getCompanyCode() {
        return companyCode;
    }

    public List<Transaction> getTransactions() {
        return transactions;
    }

    public Transaction addTransaction(Transaction transaction) {
        transactions.add(transaction);
        return transaction;
    }

    public void removeTransaction(Transaction transaction) {
        transactions.remove(transaction);
    }

    public void setCode(String companyCode) {
        this.companyCode = companyCode;
    }
}