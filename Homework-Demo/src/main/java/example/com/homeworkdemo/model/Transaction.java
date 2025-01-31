package example.com.homeworkdemo.model;

import com.fasterxml.jackson.annotation.JsonIgnore;
import jakarta.persistence.*;

import java.util.Date;

@Entity
@Table(name = "Transaction")
public class Transaction {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    Long id;

    Date date;
    String close;
    String max;
    String min;
    String avgPrice;
    String perChange;
    String volume;
    String turnoverInBest;
    String totalTurnover;
    String SMA20;
    String SMA50;
    String EMA20;
    String EMA50;
    String BBMid;
    String RSI;
    String OBV;
    String momentum;
    String CCI;
    String signal;
    String CCISignal;


    @ManyToOne
    @JoinColumn(name = "companyCode")
    @JsonIgnore
    Company company;

    public Transaction(Long id, Date date, String close, String max, String min, String avgPrice, String perChange, String volume, String turnoverInBest, String totalTurnover, String SMA20, String SMA50, String EMA20, String EMA50, String BBMid, String RSI, String OBV, String momentum, String CCI, String signal, String CCISignal) {
        this.id = id;
        this.date = date;
        this.close = close;
        this.max = max;
        this.min = min;
        this.avgPrice = avgPrice;
        this.perChange = perChange;
        this.volume = volume;
        this.turnoverInBest = turnoverInBest;
        this.totalTurnover = totalTurnover;
        this.SMA20 = SMA20;
        this.SMA50 = SMA50;
        this.EMA20 = EMA20;
        this.EMA50 = EMA50;
        this.BBMid = BBMid;
        this.RSI = RSI;
        this.OBV = OBV;
        this.momentum = momentum;
        this.CCI = CCI;
        this.signal = signal;
        this.CCISignal = CCISignal;
    }

    public Transaction() {}

    public Long getId() {
        return id;
    }

    public Date getDate() {
        return date;
    }

    public String getClose() {
        return close;
    }

    public String getMax() {
        return max;
    }

    public String getMin() {
        return min;
    }

    public String getAvgPrice() {
        return avgPrice;
    }

    public String getPerChange() {
        return perChange;
    }

    public String getVolume() {
        return volume;
    }

    public String getTurnoverInBest() {
        return turnoverInBest;
    }

    public String getTotalTurnover() {
        return totalTurnover;
    }

    public String getSMA20() {
        return SMA20;
    }

    public String getSMA50() {
        return SMA50;
    }

    public String getEMA20() {
        return EMA20;
    }

    public String getEMA50() {
        return EMA50;
    }

    public String getBBMid() {
        return BBMid;
    }

    public String getRSI() {
        return RSI;
    }

    public String getOBV() {
        return OBV;
    }

    public String getMomentum() {
        return momentum;
    }

    public String getCCI() {
        return CCI;
    }

    public String getSignal() {
        return signal;
    }

    public String getCCISignal() {
        return CCISignal;
    }

    public Company getCompany() {
        return company;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public void setDate(Date date) {
        this.date = date;
    }

    public void setClose(String close) {
        this.close = close;
    }

    public void setMax(String max) {
        this.max = max;
    }

    public void setMin(String min) {
        this.min = min;
    }

    public void setAvgPrice(String avgPrice) {
        this.avgPrice = avgPrice;
    }

    public void setPerChange(String perChange) {
        this.perChange = perChange;
    }

    public void setVolume(String volume) {
        this.volume = volume;
    }

    public void setTurnoverInBest(String turnoverInBest) {
        this.turnoverInBest = turnoverInBest;
    }

    public void setTotalTurnover(String totalTurnover) {
        this.totalTurnover = totalTurnover;
    }

    public void setSMA20(String SMA20) {
        this.SMA20 = SMA20;
    }

    public void setSMA50(String SMA50) {
        this.SMA50 = SMA50;
    }

    public void setEMA20(String EMA20) {
        this.EMA20 = EMA20;
    }

    public void setEMA50(String EMA50) {
        this.EMA50 = EMA50;
    }

    public void setBBMid(String BBMid) {
        this.BBMid = BBMid;
    }

    public void setRSI(String RSI) {
        this.RSI = RSI;
    }

    public void setOBV(String OBV) {
        this.OBV = OBV;
    }

    public void setMomentum(String momentum) {
        this.momentum = momentum;
    }

    public void setCCI(String CCI) {
        this.CCI = CCI;
    }

    public void setSignal(String signal) {
        this.signal = signal;
    }

    public void setCCISignal(String CCISignal) {
        this.CCISignal = CCISignal;
    }

    public void setCompany(Company company) {
        this.company = company;
    }
}
