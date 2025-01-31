package example.com.homeworkdemo.web;

import example.com.homeworkdemo.model.Transaction;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import example.com.homeworkdemo.service.CompanyService;
import org.springframework.web.client.RestTemplate;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "http://localhost:3000") // <-- Add this
public class Controller {
    private final CompanyService companyService;
    private final RestTemplate restTemplate = new RestTemplate();


    public Controller(CompanyService companyService) {
        this.companyService = companyService;
        this.companyService.fetchAndSaveCompany();
        this.companyService.fetchAndSaveTransaction();
    }

    @GetMapping("/get/codes")
    public ResponseEntity<List<String>> getAllCompanies() {
        return ResponseEntity.ok(companyService.getAllCodes());
    }

    @GetMapping("/get/transaction/{code}")
    public ResponseEntity<List<Transaction>> getTransactionByCode(@PathVariable String code) {
        return ResponseEntity.ok(companyService.getTransactionByCompanyCode(code));
    }
}
