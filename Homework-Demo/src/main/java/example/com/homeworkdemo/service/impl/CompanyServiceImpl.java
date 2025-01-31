package example.com.homeworkdemo.service.impl;

import example.com.homeworkdemo.model.Company;
import example.com.homeworkdemo.model.Transaction;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import example.com.homeworkdemo.repository.CompanyRepository;
import example.com.homeworkdemo.repository.TransactionRepository;
import example.com.homeworkdemo.service.CompanyService;

import java.time.LocalDateTime;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
public class CompanyServiceImpl implements CompanyService {
    @Autowired
    private final CompanyRepository companyRepository;

    @Autowired
    private final TransactionRepository transactionRepository;

    private final RestTemplate restTemplate = new RestTemplate();

    public CompanyServiceImpl(CompanyRepository companyRepository, TransactionRepository transactionRepository) {
        this.companyRepository = companyRepository;
        this.transactionRepository = transactionRepository;
    }

    @Override
    public List<Company> getAll() {
        return companyRepository.findAll();
    }

    @Override
    public Company getByCompanyCode(String code) {
        return companyRepository.findById(code).orElse(null);
    }

    @Override
    public List<Transaction> getTransactionByCompanyCode(String code) {
        Company company = companyRepository.findById(code).orElse(null);

        assert company != null;
        return company.getTransactions();
    }

    @Override
    public List<String> getAllCodes() {
        return companyRepository.findAll().stream().map(Company::getCompanyCode).collect(Collectors.toList());
    }

    @Override
    public void fetchAndSaveCompany() {
        List<String> codes = restTemplate.getForObject("http://python-service:5000/api/get/codes", List.class);
        assert codes != null;
        for (String code : codes) {
            Company company = new Company(code);
            companyRepository.save(company);
        }
    }

    @Override
    public void fetchAndSaveTransaction() {
        // For each existing company in the DB...
        for (Company company : companyRepository.findAll()) {
            String url = String.format("http://python-service:5000/api/get/transactions/%s", company.getCompanyCode());
            List<Map<String,Object>> transactions = restTemplate.getForObject(url, List.class);

            if (transactions == null || transactions.isEmpty()) {
                // No data for this company
                continue;
            }

            for (Map<String, Object> transactionMap : transactions) {
                // 1) Convert "Date" field from epoch millis to java.util.Date
                Object rawDate = transactionMap.get("Date");
                long epochMillis = ((Number) rawDate).longValue();
                Date date = new Date(epochMillis);

                // 2) Convert other fields to String
                String close          = toStringOrNull(transactionMap.get("Close"));
                String max            = toStringOrNull(transactionMap.get("Max"));
                String min            = toStringOrNull(transactionMap.get("Min"));
                String avgPrice       = toStringOrNull(transactionMap.get("Avg. Price"));
                String perChange      = toStringOrNull(transactionMap.get("%chg."));
                String volume         = toStringOrNull(transactionMap.get("Volume"));
                String turnoverInBest = toStringOrNull(transactionMap.get("Turnover in BEST"));
                String totalTurnover  = toStringOrNull(transactionMap.get("Total turnover"));
                String sma20          = toStringOrNull(transactionMap.get("SMA_20"));
                String sma50          = toStringOrNull(transactionMap.get("SMA_50"));
                String ema20          = toStringOrNull(transactionMap.get("EMA_20"));
                String ema50          = toStringOrNull(transactionMap.get("EMA_50"));
                String bbMid          = toStringOrNull(transactionMap.get("BB_Mid"));
                String rsi            = toStringOrNull(transactionMap.get("RSI"));
                String obv            = toStringOrNull(transactionMap.get("OBV"));
                String momentum       = toStringOrNull(transactionMap.get("Momentum"));
                String cci            = toStringOrNull(transactionMap.get("CCI"));
                String signal         = toStringOrNull(transactionMap.get("Signal"));
                String cciSignal      = toStringOrNull(transactionMap.get("CCI_Signal"));

                // 3) Create Transaction entity and set all fields
                Transaction transaction = new Transaction();
                transaction.setDate(date);
                transaction.setClose(close);
                transaction.setMax(max);
                transaction.setMin(min);
                transaction.setAvgPrice(avgPrice);
                transaction.setPerChange(perChange);
                transaction.setVolume(volume);
                transaction.setTurnoverInBest(turnoverInBest);
                transaction.setTotalTurnover(totalTurnover);
                transaction.setSMA20(sma20);
                transaction.setSMA50(sma50);
                transaction.setEMA20(ema20);
                transaction.setEMA50(ema50);
                transaction.setBBMid(bbMid);
                transaction.setRSI(rsi);
                transaction.setOBV(obv);
                transaction.setMomentum(momentum);
                transaction.setCCI(cci);
                transaction.setSignal(signal);
                transaction.setCCISignal(cciSignal);

                // Link the transaction to the current company
                transaction.setCompany(company);

                // 4) Save the transaction to the DB
                transactionRepository.save(transaction);
            }
        }
    }

    // Helper method to handle nulls
    private static String toStringOrNull(Object value) {
        return (value == null) ? null : value.toString();
    }

}